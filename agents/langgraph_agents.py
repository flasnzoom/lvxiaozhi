import logging
from datetime import date, datetime, timedelta
from typing import TypedDict, Annotated, Dict, List, Any, Optional
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import json
import asyncio
import sys

from config.langgraph_config import langgraph_config as config

logger = logging.getLogger(__name__)

# 常量
MAX_STEPS = 20                 # 协调员轮次上限
RECURSION_LIMIT = 40           # 图步数上限
MAX_TOOL_ROUNDS = 2            # 每个 ReAct 专家最多几轮工具往返
MAX_TOOL_CALLS_PER_ROUND = 3   # 单轮并行 tool_calls 上限
NODE_TIMEOUT = 180             # 单节点软超时（秒）
OUTPUT_SUMMARY_CHARS = 300     # 喂给协调员/汇总的摘要长度
WEATHER_MAX_DAYS = 10
WEATHER_USE_MCP = True

EXPERT_ORDER = [
    "travel_advisor",
    "weather_analyst",
    "local_expert",
    "budget_optimizer",
    "itinerary_planner",
]

# 哪些专家让模型自己决定调不调工具（react），哪些由代码先查好（prefetch）
REACT_EXPERTS: List[str] = ["travel_advisor", "local_expert"]
PREFETCH_EXPERTS: List[str] = ["weather_analyst", "budget_optimizer"]

# 每个专家能用的工具
EXPERT_TOOLS = {
    "travel_advisor": ["search_attractions", "search_destination_info"],
    "weather_analyst": ["get_daily_forecast", "get_weather_warning"],
    "budget_optimizer": ["search_budget_info", "search_hotels"],
    "local_expert": ["search_local_tips", "search_restaurants"],
    "itinerary_planner": [],
}

# 依赖申明
EXPERT_READS = {
    "local_expert": ["travel_advisor"],                    # 知道已推荐了哪些景点，避免重复/冲突
    "budget_optimizer": ["travel_advisor", "local_expert"],  # 拿真实餐饮/门票价对账
    "itinerary_planner": ["travel_advisor", "weather_analyst", "local_expert", "budget_optimizer"],
}


def _parse_date(text: Any) -> Optional[date]:
    """把字符串当日期解析，认不出来返回 None；已经是 date 就原样返回。"""
    if isinstance(text, date):
        return text
    text = str(text or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y年%m月%d日"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _normalize_dates(request: Dict[str, Any], duration: int):
    """归一成 (开始日, 结束日, 天数)。只认 start_date / end_date。"""
    start = _parse_date(request.get("start_date"))
    end = _parse_date(request.get("end_date"))

    if start is None and end is None and request.get("travel_dates"):
        logger.warning("收到未识别的 travel_dates=%r，已忽略；请改用 start_date/end_date",
                       request.get("travel_dates"))

    step = max(duration, 1) - 1
    if start is not None and end is None:
        end = start + timedelta(days=step)
    if end is not None and start is None:
        start = end - timedelta(days=step)
    if start is not None and end is not None:
        if end < start:
            start, end = end, start
        duration = (end - start).days + 1
    return start, end, duration


def _fmt_date_range(state: Dict[str, Any]) -> str:
    """给 prompt / 计划展示用的日期文案。"""
    start = state.get("start_date")
    end = state.get("end_date")
    if start is not None and end is not None:
        return str(start) + " ~ " + str(end)
    if start is not None:
        return "从 " + str(start) + " 出发"
    if end is not None:
        return "结束于 " + str(end)
    return "未指定"


def _weather_days(state: Dict[str, Any]) -> int:
    """要查几天天气 = 从今天到行程最后一天（和风上限 10 天）。"""
    end = state.get("end_date")
    if end is None:
        try:
            return max(1, min(int(state.get("duration") or 3), WEATHER_MAX_DAYS))
        except (TypeError, ValueError):
            return 3
    days = (end - date.today()).days + 1
    if days < 1:
        return 1
    return min(days, WEATHER_MAX_DAYS)


# 兴趣约束
def _fmt_interests(interests: Any) -> str:
    if isinstance(interests, str):
        return interests or "无特殊偏好"
    if isinstance(interests, (list, tuple)):
        return "、".join(str(i) for i in interests) or "无特殊偏好"
    return "无特殊偏好"


# 控制上下文，截断长文本
def _brief(text: Any, limit: int = OUTPUT_SUMMARY_CHARS) -> str:
    if text is None:
        return ""
    s = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)
    s = s.strip().replace("\n", " ")
    return s if len(s) <= limit else s[:limit] + "…"


def create_initial_state(request: Dict[str, Any]) -> Dict[str, Any]:
    duration = 3
    try:
        duration = max(1, int(request.get("duration") or 3))
    except (TypeError, ValueError):
        duration = 3

    start, end, duration = _normalize_dates(request, duration)

    return {
        "messages": [],
        "destination": request.get("destination", ""),
        "duration": duration,
        "budget_range": request.get("budget_range", ""),
        "interests": list(request.get("interests", []) or []),
        "group_size": int(request.get("group_size", 1) or 1),
        "start_date": start,
        "end_date": end,
        "next_action": "",
        "iteration_count": 0,
        "agent_outputs": {},
        "final_plan": {},
    }


# 定义多智能体的状态结构
class TravelPlanState(TypedDict):
    """
    旅行规划状态类

    这个类定义了所有智能体共享的状态结构，包含了
    旅行规划过程中需要的所有信息。

    属性说明：
    - messages: 智能体间的消息历史
    - destination: 目的地
    - duration: 旅行天数
    - budget_range: 预算范围
    - interests: 兴趣爱好列表
    - group_size: 团队人数
    - start_date: 旅行开始日期
    - end_date: 旅行结束日期
    - next_action: 协调员的调度结果
    - iteration_count: 迭代次数
    - agent_outputs: 各智能体的输出结果
    - final_plan: 最终的旅行计划
    """

    messages: Annotated[List[AnyMessage], add_messages]
    # 请求参数
    destination: str
    duration: int
    budget_range: str
    interests: List[str]
    group_size: int
    start_date: Optional[date]
    end_date: Optional[date]
    # 调度
    next_action: str
    iteration_count: int
    # 产出
    agent_outputs: Dict[str, Any]
    final_plan: Dict[str, Any]


class LangGraphTravelAgents:
    """多智能体系统"""

    def __init__(self, llm: Any = None, tools: Dict[str, Any] = None):
        # 初始化模型
        if llm is None:
            llm = ChatOpenAI(**config.get_llm_config())
        self.llm = llm

        # 注册工具
        if tools is None:
            self.tools = self._register_local_tools()
        else:
            self.tools = dict(tools)

        self.tool_llms: Dict[str, Any] = self.bind_tools_to_experts()

        # 初始化工作流图
        self.graph = self._create_agent_graph()

    async def setup_mcp(self) -> None:
        if not WEATHER_USE_MCP:
            return
        from tools.mcp_weather_client import fetch_weather_tools
        mcp_tools = await fetch_weather_tools()
        self.tools.update(mcp_tools)  # 同名覆盖 get_daily_forecast
        self.tool_llms = self.bind_tools_to_experts()  # 必须重绑
        logger.info("MCP 工具已接管: %s（总数 %s）", sorted(mcp_tools), len(self.tools))

    # 工具注册
    def _register_local_tools(self) -> Dict[str, BaseTool]:
        from tools import travel_tools

        tools: Dict[str, BaseTool] = {}
        for name, value in vars(travel_tools).items():
            if isinstance(value, BaseTool):
                tools[value.name] = value

        logger.info("本地工具 %s 个: %s", len(tools), sorted(tools))
        return tools

    def bind_tools_to_experts(self) -> Dict[str, Any]:
        """只给 react 专家绑工具；其余专家不需要，值是 None。"""
        result: Dict[str, Any] = {}
        for name in EXPERT_ORDER:
            if name not in REACT_EXPERTS:
                result[name] = None
                continue

            tools_for_this_expert: List[BaseTool] = []
            for tool_name in EXPERT_TOOLS[name]:
                if tool_name in self.tools:
                    tools_for_this_expert.append(self.tools[tool_name])

            if tools_for_this_expert:
                result[name] = self.llm.bind_tools(tools_for_this_expert)
            else:
                result[name] = None
        return result

    async def call_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """执行一个工具。工具不存在或报错，都返回一句话说明，不往上抛。"""
        tool = self.tools.get(tool_name)
        if tool is None:
            return "（工具 " + tool_name + " 不可用）"
        try:
            result = await tool.ainvoke(args)
            return str(result)
        except Exception as exc:
            logger.exception("工具 %s 执行失败", tool_name)
            return "工具执行失败：" + str(exc)

    def _create_agent_graph(self) -> StateGraph:
        """创建多智能体工作流图"""
        # 定义工作流图
        workflow = StateGraph(TravelPlanState)

        # 添加节点
        workflow.add_node("travel_advisor", self._travel_advisor_node)
        workflow.add_node("weather_analyst", self._weather_analyst_node)
        workflow.add_node("budget_optimizer", self._budget_optimizer_node)
        workflow.add_node("local_expert", self._local_expert_node)
        workflow.add_node("itinerary_planner", self._itinerary_planner_node)
        workflow.add_node("coordinator", self._coordinator_node)
        workflow.add_node("plan_writer", self._plan_writer_node)

        # 定义工作流边缘
        workflow.set_entry_point("coordinator")  # 入口
        workflow.add_conditional_edges("coordinator", self.coordinator_router)

        # 每个专家干完活都回到协调员排队
        workflow.add_edge("travel_advisor", "coordinator")
        workflow.add_edge("weather_analyst", "coordinator")
        workflow.add_edge("local_expert", "coordinator")
        workflow.add_edge("budget_optimizer", "coordinator")
        workflow.add_edge("itinerary_planner", "coordinator")
        workflow.add_edge("plan_writer", END)

        # 编译返回工作流
        return workflow.compile()

    def pick_next_expert(self, state: TravelPlanState) -> Optional[str]:
        """按顺序找第一个还没做完的专家；都做完了返回 None。"""
        outputs = state.get("agent_outputs") or {}
        for name in EXPERT_ORDER:
            info = outputs.get(name) or {}
            if info.get("status") not in ("completed", "failed"):
                return name
        return None

    async def _coordinator_node(self, state: TravelPlanState) -> Dict[str, Any]:
        count = state.get("iteration_count", 0) + 1

        if count > MAX_STEPS:
            logger.warning("转圈超过 %s 次，强制结束", MAX_STEPS)
            return {"next_action": "end", "iteration_count": count}

        nxt = self.pick_next_expert(state)
        if nxt is None:
            return {"next_action": "plan_writer", "iteration_count": count}
        return {"next_action": nxt, "iteration_count": count}

    def coordinator_router(self, state: TravelPlanState) -> str:
        """把 next_action 变成本次要走的边。"""
        action = state.get("next_action") or ""
        if action in EXPERT_ORDER:
            return action
        if action == "plan_writer":
            return "plan_writer"
        return END

    # 5个节点
    async def _travel_advisor_node(self, state: TravelPlanState) -> Dict[str, Any]:
        return await self.run_expert(state, "travel_advisor")

    async def _weather_analyst_node(self, state: TravelPlanState) -> Dict[str, Any]:
        return await self.run_expert(state, "weather_analyst")

    async def _local_expert_node(self, state: TravelPlanState) -> Dict[str, Any]:
        return await self.run_expert(state, "local_expert")

    async def _budget_optimizer_node(self, state: TravelPlanState) -> Dict[str, Any]:
        return await self.run_expert(state, "budget_optimizer")

    async def _itinerary_planner_node(self, state: TravelPlanState) -> Dict[str, Any]:
        return await self.run_expert(state, "itinerary_planner")

    async def run_expert(self, state: TravelPlanState, name: str) -> Dict[str, Any]:
        """节点入口：只管超时和兜错，真正的活交给 _run_expert_inner。"""
        try:
            return await asyncio.wait_for(
                self._run_expert_inner(state, name),
                timeout=NODE_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.error("[%s] 节点超时（%s 秒），标记 failed 继续下一个", name, NODE_TIMEOUT)
            return self._pack_failed(state, name, "（" + name + " 超时未完成，本版计划不含该维度）")
        except Exception as exc:
            logger.exception("[%s] 节点异常，标记 failed 继续下一个", name)
            return self._pack_failed(state, name, "（" + name + " 执行失败：" + str(exc) + "）")

    def _pack_failed(self, state: TravelPlanState, name: str, note: str) -> Dict[str, Any]:
        """失败也返回 patch：不写 messages（手里没有合法的 message 对象），只写 agent_outputs。"""
        outputs: Dict[str, Any] = dict(state.get("agent_outputs") or {})
        outputs[name] = {"response": note, "status": "failed"}
        return {"agent_outputs": outputs}

    async def _run_expert_inner(self, state: TravelPlanState, name: str) -> Dict[str, Any]:
        """模式分流"""
        messages: List[AnyMessage] = [SystemMessage(content=self.build_prompt(state, name))]

        if name in REACT_EXPERTS:
            response = await self.step_with_tools(messages, name)
        elif name in PREFETCH_EXPERTS:
            response = await self.step_with_prefetch(messages, state, name)
        else:
            response = await self.llm.ainvoke(messages)

        text: str = response.content
        if not isinstance(text, str):
            text = str(text)

        outputs: Dict[str, Any] = dict(state.get("agent_outputs") or {})
        outputs[name] = {"response": text, "status": "completed"}

        return {"messages": [response], "agent_outputs": outputs}

    async def step_with_tools(self, messages: List[AnyMessage], name: str) -> Any:
        """react模式"""
        llm_with_tools = self.tool_llms.get(name)
        if llm_with_tools is None:
            return await self.llm.ainvoke(messages)

        response = await llm_with_tools.ainvoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return response  # 模型觉得不用查，直接答了

        for call in response.tool_calls:
            result: str = await self.call_tool(call["name"], call["args"])
            messages.append(
                ToolMessage(content=result, tool_call_id=call["id"], name=call["name"])
            )

        return await self.llm.ainvoke(messages)  # 拿着工具结果再问一次

    async def step_with_prefetch(self, messages: List[AnyMessage], state: TravelPlanState, name: str) -> Any:
        """prefetch模式"""
        data: str

        if name == "weather_analyst":
            days = _weather_days(state)
            data = await self.call_tool(
                "get_daily_forecast",
                {"location": state.get("destination", ""), "days": days},
            )
            today = date.today()
            last = today + timedelta(days=days - 1)
            note = "（本次天气数据覆盖 " + str(today) + " 至 " + str(last) + "；"
            if state.get("end_date") is None:
                note += "行程日期未指定，这只是近期趋势，不要当成行程期预报）\n"
            else:
                note += "行程期是 " + _fmt_date_range(state) + "，覆盖不到的日期请写“待确认”）\n"
            data = note + data
        else:  # budget_optimizer
            part1: str = await self.call_tool(
                "search_budget_info",
                {"destination": state.get("destination", ""),
                 "duration": str(state.get("duration", ""))},
            )
            part2: str = await self.call_tool(
                "search_hotels",
                {"destination": state.get("destination", ""),
                 "budget": state.get("budget_range") or "中等预算"},
            )
            data = part1 + "\n\n" + part2

        messages.append(HumanMessage(content="[系统预先查询到的数据]\n" + data))
        return await self.llm.ainvoke(messages)

    def build_prompt(self, state: TravelPlanState, name: str) -> str:
        if name == "travel_advisor":
            return self.prompt_travel_advisor(state)
        if name == "weather_analyst":
            return self.prompt_weather_analyst(state)
        if name == "local_expert":
            return self.prompt_local_expert(state)
        if name == "budget_optimizer":
            return self.prompt_budget_optimizer(state)
        return self.prompt_itinerary_planner(state)

    def build_evidence(self, state: TravelPlanState, names: List[str], limit: int = OUTPUT_SUMMARY_CHARS) -> str:
        """把上游专家的结论拼成一段文字。没有就返回一句说明。"""
        outputs = state.get("agent_outputs") or {}
        parts: List[str] = []
        for upstream in names:
            info = outputs.get(upstream) or {}
            if info.get("status") == "failed":
                continue
            text = _brief(info.get("response"), limit)
            if text:
                parts.append("### " + upstream + "\n" + text)
        if not parts:
            return "（暂无上游产出）"
        return "\n\n".join(parts)

    def prompt_travel_advisor(self, state: TravelPlanState) -> str:
        return (
            "您是旅行顾问智能体。\n\n"
            "当前请求：\n"
            "- 目的地: " + str(state.get("destination")) + "\n"
            "- 时长: " + str(state.get("duration")) + " 天\n"
            "- 兴趣: " + _fmt_interests(state.get("interests")) + "\n"
            "- 团队人数: " + str(state.get("group_size")) + "\n\n"
            "需要实时信息（门票、开放情况）时可以调用工具。\n"
            "任务：1) 必游景点 2) 文化礼仪 3) 住宿区域 4) 兴趣相关活动。800-1200 字。\n"
            "不要编造具体票价，没有数据就写“参考区间，建议现场确认”。"
        )

    def prompt_weather_analyst(self, state: TravelPlanState) -> str:
        return (
            "您是天气分析师智能体。\n\n"
            "当前请求：\n"
            "- 目的地: " + str(state.get("destination")) + "\n"
            "- 旅行日期: " + _fmt_date_range(state) + "\n"
            "- 时长: " + str(state.get("duration")) + " 天\n\n"
            "如果下面提供了天气数据，就基于真实数据分析：逐日天气、户外最佳时段、"
            "室内备选、穿衣建议。没有数据就写“天气数据待确认”，不要编造温度。"
        )

    def prompt_local_expert(self, state: TravelPlanState) -> str:
        evidence: str = self.build_evidence(state, EXPERT_READS["local_expert"])
        return (
            "您是当地专家智能体。\n\n"
            "当前请求：\n"
            "- 目的地: " + str(state.get("destination")) + "\n"
            "- 兴趣: " + _fmt_interests(state.get("interests")) + "\n\n"
            "## 上游已推荐的内容（别重复，优先补充小众/替代项）\n"
            + evidence + "\n\n"
            "任务：1) 小众去处 2) 当地习俗禁忌 3) 本地人怎么吃（带人均） 4) 避坑。"
        )

    def prompt_budget_optimizer(self, state: TravelPlanState) -> str:
        evidence: str = self.build_evidence(state, EXPERT_READS["budget_optimizer"], limit=600)
        return (
            "您是预算优化师智能体。\n\n"
            "当前请求：\n"
            "- 目的地: " + str(state.get("destination")) + "\n"
            "- 时长: " + str(state.get("duration")) + " 天\n"
            "- 预算范围: " + str(state.get("budget_range") or "未指定") + "\n"
            "- 团队人数: " + str(state.get("group_size")) + "\n\n"
            "## 上游已获取的信息（可能含真实价格）\n"
            + evidence + "\n\n"
            "请给出：1) 人均/总预算 2) 分项估算 3) 省钱策略 4) 超预算替代方案。\n"
            "若上游价格与你的估算差超过 20%，单独说明原因，不要静默覆盖。"
        )

    def prompt_itinerary_planner(self, state: TravelPlanState) -> str:
        evidence: str = self.build_evidence(state, EXPERT_READS["itinerary_planner"], limit=600)
        return (
            "您是行程规划师智能体，负责把专家结论编排成逐日行程。\n\n"
            "请求：" + str(state.get("destination")) + " / " + str(state.get("duration"))
            + " 天 / " + str(state.get("group_size")) + " 人\n\n"
            "其他专家结论：\n" + evidence + "\n\n"
            "要求：按天编排（上午/下午/晚上），考虑地理位置就近、天气影响、用餐时间。\n"
            "不要罗列原文，要给出可执行的顺序。"
        )

    async def _plan_writer_node(self, state: TravelPlanState) -> Dict[str, Any]:
        outputs = state.get("agent_outputs") or {}
        parts: List[str] = []
        for name in EXPERT_ORDER:
            info = outputs.get(name) or {}
            if info.get("status") != "completed":
                continue
            parts.append("### " + name + "\n" + _brief(info.get("response"), 1500))

        digest: str
        if not parts:
            digest = "（没有专家产出）"
        else:
            digest = "\n\n".join(parts)

        prompt = (
            "你是首席旅行规划师，把专家结论整合成一份完整旅行计划。\n\n"
            "请求：" + str(state.get("destination")) + " / " + str(state.get("duration")) + "天 / "
            + str(state.get("group_size")) + "人 / 预算 "
            + str(state.get("budget_range") or "未指定") + "\n\n"
            "专家结论：\n" + digest + "\n\n"
            "要求：七个章节（行程概览/日程安排/预算估算/住宿推荐/交通指南/美食推荐/实用贴士）。\n"
            "数字冲突时以有工具数据来源的为准；缺数据写“待确认”，不要编。只输出 Markdown。"
        )

        response = await self.llm.ainvoke([SystemMessage(content=prompt)])
        content: str = response.content
        if not isinstance(content, str):
            content = str(content)

        plan: Dict[str, Any] = {
            "destination": state.get("destination"),
            "duration": state.get("duration"),
            "budget_range": state.get("budget_range"),
            "interests": state.get("interests"),
            "group_size": state.get("group_size"),
            "travel_dates": _fmt_date_range(state),
            "content": content,
        }
        return {"final_plan": plan}

    async def run_travel_planning(self, request: Dict[str, Any]) -> Dict[str, Any]:
        state = create_initial_state(request)
        try:
            final = await self.graph.ainvoke(state, config={"recursion_limit": RECURSION_LIMIT})
        except Exception as exc:
            logger.exception("规划失败")
            return {"success": False, "error": str(exc), "travel_plan": {}}

        plan = final.get("final_plan") or {}
        return {
            "success": bool(plan),
            "travel_plan": plan,
            "agent_outputs": final.get("agent_outputs", {}),
            "total_iterations": final.get("iteration_count", 0),
        }


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    from agents.request_extractor import extract, ask_for_more

    query = "下个周末去成都玩3天，两个人，预算三千到五千，想吃美食和看古迹"
    logger.info("用户输入：%s", query)

    request = extract(query)
    logger.info("提取结果：%s", request)

    if request["missing"]:
        print("还缺信息，需要追问：")
        print(ask_for_more(request["missing"]))
        sys.exit(0)

    request.pop("missing", None)

    agent = LangGraphTravelAgents()

    async def _run() -> dict:
        await agent.setup_mcp()
        return await agent.run_travel_planning(request)

    result = asyncio.run(_run())
    print("success:", result["success"], "\n", result["travel_plan"])
