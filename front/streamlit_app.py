import os
import time

import requests
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

# 三个超时
CHAT_TIMEOUT = 30
PLAN_TIMEOUT = 10
POLL_INTERVAL = 2        # 轮询间隔
POLL_DEADLINE = 240

st.set_page_config(page_title="旅小智 · 旅行规划", page_icon="🧭", layout="centered")


# ---------------------------------------------------------------- 状态
def init_state() -> None:
    """只在第一次进来时初始化；之后每次 rerun 都复用。"""
    if "messages" not in st.session_state:
        st.session_state.messages = []      # [(role, kind, payload), ...]
    if "session_id" not in st.session_state:
        st.session_state.session_id = None  # 第一轮由后端发，之后每轮带上
    if "task_id" not in st.session_state:
        st.session_state.task_id = None


def reset_session() -> None:
    """新开一段对话：清掉本地记录，session_id 置空。"""
    st.session_state.messages = []
    st.session_state.session_id = None
    st.session_state.task_id = None


# ---------------------------------------------------------------- 后端调用
def post_chat(message: str) -> dict:
    payload = {"message": message}
    if st.session_state.session_id:
        payload["session_id"] = st.session_state.session_id
    resp = requests.post(API_BASE + "/chat", json=payload, timeout=CHAT_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def get_plan(task_id: str) -> dict:
    resp = requests.get(API_BASE + "/plan/" + task_id, timeout=PLAN_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------- 渲染
def render_plan(plan: dict, key: str) -> None:
    """计划气泡：抬头指标 → 假设说明 → 正文 → 下载。"""
    if not plan:
        st.warning("没有拿到计划内容，请重新提问。")
        return

    col1, col2 = st.columns(2)
    col1.metric("目的地", plan.get("destination") or "未填")
    col2.metric("天数", "%s 天" % (plan.get("duration") or "?"))

    col3, col4 = st.columns(2)
    people = plan.get("group_size")
    col3.metric("人数", ("%s 人" % people) if people else "按 1 人估算")
    budget = plan.get("budget_range") or ""
    if budget and budget.startswith("0-"):
        budget_text = "≤ %s 元" % budget[2:]
    elif budget:
        budget_text = budget + " 元"
    else:
        budget_text = "未指定"
    col4.metric("预算", budget_text)

    dates = plan.get("travel_dates")
    if dates:
        st.caption("行程日期：" + str(dates))

    # M5 验收时发现的坑：人数/预算缺省是后端兜的，用户不知道，必须明说
    assumptions = []
    if not people:
        assumptions.append("按 1 人估算")
    if not plan.get("budget_range"):
        assumptions.append("未指定预算，费用为参考值")
    if assumptions:
        st.info("假设：" + "；".join(assumptions))

    content = plan.get("content") or ""
    st.markdown(content)

    if content:
        st.download_button(
            label="下载 Markdown",
            data=content,
            file_name=make_filename(plan),
            mime="text/markdown",
            key="download-" + key,
        )


def make_filename(plan: dict) -> str:
    dest = str(plan.get("destination") or "旅行")
    days = plan.get("duration")
    safe = dest.replace("/", "-").replace("\\", "-").replace(":", "-")
    return "%s-%s天-旅行计划.md" % (safe, days if days else "?") if days else safe + "-旅行计划.md"


def render_history() -> None:
    """每条历史重画一次；计划类消息重新走 render_plan。"""
    for index, item in enumerate(st.session_state.messages):
        role, kind, payload = item
        with st.chat_message(role):
            if kind == "plan":
                render_plan(payload, key="h%d" % index)
            else:
                st.markdown(payload)


# ---------------------------------------------------------------- 轮询
def poll_plan(box) -> dict:
    """同步轮询，进度写进传入的占位符。返回 plan；失败/超时返回 {}。"""
    task_id = st.session_state.task_id
    started = time.time()

    while time.time() - started < POLL_DEADLINE:
        try:
            data = get_plan(task_id)
        except requests.RequestException as exc:
            box.warning("查询进度失败，正在重试…（%s）" % exc)
            time.sleep(POLL_INTERVAL)
            continue

        status = data.get("status")
        if status == "done":
            version = data.get("version") or 1
            box.success("规划完成（第 %s 版）" % version)
            return data.get("plan") or {}
        if status == "failed":
            box.error("规划失败：%s" % (data.get("error") or "未知错误"))
            return {}

        waited = int(time.time() - started)
        box.info("规划中… 已等 %d 秒。通常要 2–3 分钟，别刷新页面。" % waited)
        time.sleep(POLL_INTERVAL)

    box.warning("等了 %d 秒还没出结果。任务可能仍在后台跑，稍后再问一次就能拿到新版本。" % POLL_DEADLINE)
    return {}


# ---------------------------------------------------------------- 输入
def ask_assistant(message: str) -> None:
    st.session_state.messages.append(("assistant", "text", message))
    with st.chat_message("assistant"):
        st.markdown(message)


def handle_input() -> None:
    prompt = st.chat_input("想去哪儿、玩几天？比如：去成都玩3天，两个人，预算四千")
    if not prompt:
        return

    st.session_state.messages.append(("user", "text", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # ---------- ① 发请求（网络错误要变成人话，不是 traceback）
    try:
        data = post_chat(prompt)
    except requests.RequestException as exc:
        ask_assistant(
            "连不上后端（%s）。确认 uvicorn 已在 8000 端口启动，再试一次。" % exc.__class__.__name__
        )
        return

    if data.get("session_id"):
        st.session_state.session_id = data["session_id"]
    status = data.get("status")
    version = data.get("version") or 0

    # ---------- ② 正常追问：这不是错误，显示 question 就行
    if status == "need_info":
        ask_assistant(data.get("question") or "还需要补充一点信息。")
        return

    # ---------- ③ 提取器故障 / 系统忙
    if status == "error":
        ask_assistant("⚠️ " + (data.get("question") or "解析服务暂时不可用，请稍后重试。"))
        return

    if status == "busy":
        ask_assistant("⏳ " + (data.get("question") or "上一个规划还在跑，请稍等。"))
        return

    # ---------- ④ 起任务：进度 + 轮询
    if status == "planning":
        st.session_state.task_id = data.get("task_id")
        notice = data.get("notice")

        with st.chat_message("assistant"):
            if notice:
                st.warning(notice)
            st.markdown("收到，开始规划（第 %s 版）。" % version)
            box = st.empty()
            box.info("规划中… 正在排队")
            plan = poll_plan(box)

            if plan:
                render_plan(plan, key="new%s" % version)
                st.session_state.messages.append(("assistant", "plan", plan))
            else:
                st.session_state.messages.append(
                    ("assistant", "text", "这次没拿到计划，稍后再发一次试试。")
                )
        return

    ask_assistant("后端返回了没见过的状态：%s" % status)


# ---------------------------------------------------------------- 侧边栏
def render_sidebar() -> None:
    with st.sidebar:
        st.subheader("会话")
        sid = st.session_state.session_id
        st.caption("session_id：" + (sid[:8] + "…" if sid else "（未创建）"))

        task_id = st.session_state.task_id
        st.caption("当前任务：" + (task_id[:8] + "…" if task_id else "无"))

        if st.button("开始新对话", use_container_width=True):
            reset_session()
            st.rerun()

        st.divider()
        st.caption("后端：" + API_BASE)
        try:
            healthy = requests.get(API_BASE + "/health", timeout=5).status_code == 200
        except requests.RequestException:
            healthy = False
        st.caption("状态：" + ("✅ 在线" if healthy else "❌ 连不上"))


# ---------------------------------------------------------------- 入口
def main() -> None:
    init_state()

    st.title("🧭 旅小智 · 旅行规划")
    st.caption("一句话说清想去哪儿、玩几天，剩下的交给多智能体。")

    render_sidebar()
    render_history()
    handle_input()


main()
