import asyncio
import logging
from datetime import date
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from config.langgraph_config import LangGraphConfig

logger = logging.getLogger(__name__)

WEEKDAY_NAMES = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]


class TravelRequest(BaseModel):
    destination: Optional[str] = None
    duration: Optional[int] = None
    budget_range: Optional[str] = None
    interests: List[str] = Field(default_factory=list)
    group_size: Optional[int] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None


def build_system_prompt(today: date) -> str:
    weekday = WEEKDAY_NAMES[today.weekday()]
    lines = [
        "你是旅行需求解析器。把用户的一句话解析成一个 json 对象，只输出 json，不要任何解释。",
        "",
        "今天是 " + str(today) + " " + weekday + "。所有相对日期都要基于今天换算成绝对日期"
        "（例如「这周末」「下个月第一个周末」「下周三」）。",
        "",
        "json 里只允许出现这 7 个键：",
        "- destination：目的地**城市名**（地级市或县级市），字符串；没提到就 null",
        "- duration：旅行天数，整数；「三天」=3，「一周」=7；没提到就 null",
        "- budget_range：预算，统一成「最小-最大」的人民币数字字符串，例如 \"3000-5000\"；"
        "只给上限写 \"0-上限\"，只给下限写 \"下限以上\"；「不设上限」就 null",
        "- interests：兴趣标签列表，用用户的原词，0 到 6 个；没有就 []",
        "- group_size：出行人数，整数；「两个人」=2，「一家三口」=3；"
        "用户没有明确说人数就 null，不要猜",
        "- start_date：出发日期，格式 YYYY-MM-DD；没提到就 null",
        "- end_date：返回日期，格式 YYYY-MM-DD；没提到就 null",
        "",
        "规则：",
        "1. 用户没说的信息一律 null 或 []，禁止编造，禁止用「常见值」补齐。",
        "2. 只给了天数、没给日期时，start_date 和 end_date 都填 null。",
        "3. 只给了日期、没给天数时，duration 填 null，不要自己算。",
        "4. 和旅行规划无关的话题（问天气、闲聊、写报告等），7 个键全部 null 或 []。",
        "5. 金额说「左右」「大概」时给合理区间，例如「五千左右」-> \"4500-5500\"。",
        "6. destination 必须是地级市或县级市名称。用户说的是景区、乐园、山、湖、古镇时，"
        "填它所属的城市，并把景区名原样放进 interests："
        "「西湖」-> 杭州 / 「迪士尼」-> 上海 / 「九寨沟」-> 阿坝 / 「故宫」-> 北京。",
    ]
    return "\n".join(lines)


def build_llm() -> ChatOpenAI:
    cfg = LangGraphConfig.get_llm_config()
    cfg["temperature"] = 0
    return ChatOpenAI(**cfg)


def _to_missing(req: TravelRequest) -> List[str]:
    missing: List[str] = []
    if not req.destination:
        missing.append("destination")
    if req.start_date is None and req.duration is None:
        missing.append("dates_or_duration")
    return missing


QUESTIONS = {
    "destination": "请问想去哪个城市或景区？",
    "dates_or_duration": "计划哪天出发、玩几天？（例如「这周末两天」或「11月20号到22号」）",
}


def ask_for_more(missing: List[str]) -> str:
    """把 missing 变成一句人话追问。模板化，不用模型写。"""
    parts: List[str] = []
    for key in missing:
        text = QUESTIONS.get(key)
        if text:
            parts.append(text)
    return "".join(parts)


def to_request_dict(req: TravelRequest) -> Dict[str, Any]:
    return {
        "destination": req.destination,
        "duration": req.duration,
        "budget_range": req.budget_range,
        "interests": list(req.interests),
        "group_size": req.group_size,
        "start_date": req.start_date,
        "end_date": req.end_date,
        "missing": _to_missing(req),
    }


RETRY_HINT = (
    "上一次的输出不是合法 json。请只输出 json：duration 和 group_size 必须是整数，"
    "start_date / end_date 必须是 YYYY-MM-DD 格式的字符串。"
)


async def aextract(
    query: str,
    llm: Optional[ChatOpenAI] = None,
    today: Optional[date] = None,
) -> Dict[str, Any]:
    query = (query or "").strip()
    today = today or date.today()
    empty = to_request_dict(TravelRequest())
    if not query:
        return empty

    try:
        client = llm or build_llm()
        structured = client.with_structured_output(TravelRequest, method="json_mode")
    except Exception as exc:
        logger.exception("提取器初始化失败")
        return {**empty, "error": str(exc)}

    messages = [
        SystemMessage(content=build_system_prompt(today)),
        HumanMessage(content=query),
    ]

    last_error = ""
    for attempt in (1, 2):
        try:
            req = await structured.ainvoke(messages)
            result = to_request_dict(req)
            logger.info("提取成功（第 %s 次）：%s", attempt, result)
            return result
        except Exception as exc:  # 校验失败 / 解析失败 / 网络错误都走这里
            last_error = str(exc)
            logger.warning("第 %s 次解析失败：%s", attempt, last_error)
            messages.append(HumanMessage(content=RETRY_HINT))

    logger.error("两次都失败，返回空请求（交给上游追问）：%s", last_error)
    return {**empty, "error": last_error}


def extract(
    query: str,
    llm: Optional[ChatOpenAI] = None,
    today: Optional[date] = None,
) -> Dict[str, Any]:
    """同步入口，命令行/评估集用。M5 里请直接用 aextract。"""
    return asyncio.run(aextract(query, llm=llm, today=today))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    demo = "下个月第一个周末去杭州，两个人，预算五千，想吃本地菜"
    print("输出：", extract(demo))