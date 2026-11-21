from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import logging
import asyncio
import time
from datetime import date, timedelta
from fastapi import FastAPI
from contextlib import asynccontextmanager
import uuid

from agents.request_extractor import TravelRequest, _to_missing, aextract, ask_for_more


SESSION_TTL = 3600          # 会话 1 小时过期
SESSION_LIMIT = 200         # 最多留 200 个会话
TASK_LIMIT = 500            # 最多 500 个任务
TOO_FAR_DAYS = 9            # 行程末日超出 today+9 天,天气为空
FIELDS = ("destination", "duration", "budget_range", "interests",
          "group_size", "start_date", "end_date")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("api")

sessions: Dict[str, Dict[str, Any]] = {}
tasks: Dict[str, Dict[str, Any]] = {}
graph_lock = asyncio.Semaphore(1)
agent = None


def new_session() -> Dict[str, Any]:
    empty: Dict[str, Any] = {}
    for key in FIELDS:
        empty[key] = [] if key == "interests" else None
    return {"request": empty, "missing": [], "version": 0,
            "planning_task": None, "updated_at": time.time()}


def cleanup_expired() -> None:
    """惰性清理"""
    now = time.time()
    for sid in list(sessions.keys()):
        if now - sessions[sid]["updated_at"] > SESSION_TTL:
            sessions.pop(sid, None)
    for tid in list(tasks.keys()):
        if now - tasks[tid]["updated_at"] > SESSION_TTL:
            tasks.pop(tid, None)
    if len(sessions) > SESSION_LIMIT:
        ordered = sorted(sessions.items(), key=lambda kv: kv[1]["updated_at"])
        for sid, _ in ordered[: len(sessions) - SESSION_LIMIT]:
            sessions.pop(sid, None)
    if len(tasks) > TASK_LIMIT:
        ordered = sorted(tasks.items(), key=lambda kv: kv[1]["updated_at"])
        for tid, task in ordered:
            if len(tasks) <= TASK_LIMIT:
                break
            if task["status"] != "running":
                tasks.pop(tid, None)


def merge_request(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """多轮合并"""
    merged: Dict[str, Any] = {}
    for key in FIELDS:
        merged[key] = old.get(key)

    for key in FIELDS:
        if key == "interests":
            continue
        value = new.get(key)
        if value is None or value == "":
            continue
        merged[key] = value

    interests = list(merged.get("interests") or [])
    for item in (new.get("interests") or []):
        if item not in interests:
            interests.append(item)
    merged["interests"] = interests

    start = merged.get("start_date")
    end = merged.get("end_date")
    duration = merged.get("duration")

    new_start = new.get("start_date")
    new_end = new.get("end_date")

    if new_start is not None and new_end is not None:
        # 更新出发结束
        merged["duration"] = (new_end - new_start).days + 1
    elif new_end is not None:
        # 只更新结束
        if duration is not None:
            merged["start_date"] = new_end - timedelta(days=max(int(duration), 1) - 1)
    elif new_start is not None:
        # 只更新出发
        if duration is not None:
            merged["end_date"] = new_start + timedelta(days=max(int(duration), 1) - 1)
    elif duration is not None:
        # 只改天数
        if start is not None:
            merged["end_date"] = start + timedelta(days=max(int(duration), 1) - 1)
        elif end is not None:
            merged["start_date"] = end - timedelta(days=max(int(duration), 1) - 1)
    return merged


def compute_missing(request: Dict[str, Any]) -> List[str]:
    """复用提取器的判定规则"""
    data: Dict[str, Any] = {}
    for key in FIELDS:
        data[key] = request.get(key)
    travel = TravelRequest(**data)
    return _to_missing(travel)


def too_far_notice(request: Dict[str, Any]) -> Optional[str]:
    end = request.get("end_date")
    if not isinstance(end, date):
        return None
    gap = (end - date.today()).days
    if gap > TOO_FAR_DAYS:
        return ("行程末日距今 " + str(gap) + " 天，超出 10 天预报范围："
                "天气部分只会给气候参考，出发前 1-2 天请再查一次。")
    return None


async def run_planning(task_id: str, session_id: str, request: Dict[str, Any]) -> None:
    task = tasks[task_id]
    task["status"] = "running"
    session = sessions.get(session_id)
    try:
        async with graph_lock:          # 同一时刻只跑一个图
            result = await agent.run_travel_planning(request)
            if result.get("success"):
                task["plan"] = result.get("travel_plan") or {}
                task["status"] = "done"
            else:
                task["error"] = result.get("error") or "规划失败"
                task["status"] = "failed"
    except Exception as exc:
        logger.exception("后台任务异常")
        task["error"] = str(exc)
        task["status"] = "failed"
    finally:
        task["updated_at"] = time.time()
        if session is not None and session.get("planning_task") == task_id:
            session["planning_task"] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    from agents.langgraph_agents import LangGraphTravelAgents
    agent = LangGraphTravelAgents()  # 全局单例：构造时会注册 7 个工具
    await agent.setup_mcp()
    logger.info("图已加载")
    yield


app = FastAPI(title="旅行规划 API", lifespan=lifespan)


class ChatIn(BaseModel):
    session_id: Optional[str] = None
    message: str


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    session = sessions.get(session_id)
    if session is None:
        return {"found": False}
    return {"found": True, "request": session["request"],
            "missing": session["missing"], "version": session["version"],
            "planning": session["planning_task"] is not None}


@app.post("/chat")
async def chat(payload: ChatIn):
    cleanup_expired()

    session_id = payload.session_id or uuid.uuid4().hex
    session = sessions.get(session_id)
    if session is None:
        session = new_session()
        sessions[session_id] = session

    # ① 忙就不干活
    if session["planning_task"] is not None or session.get("busy"):
        return {"session_id": session_id, "status": "busy", "version": session["version"],
                "question": "上一次规划还在跑（约 2 分钟），请稍后再发。"}
    if graph_lock.locked():
        return {"session_id": session_id, "status": "busy", "version": session["version"],
                "question": "系统正在处理另一个请求，请稍后再试。"}

    message = (payload.message or "").strip()
    if not message:
        return {"session_id": session_id, "status": "need_info", "version": session["version"],
                "question": "请描述一下想去哪里、玩几天。"}

    session["busy"] = True
    try:
        # ② 提取
        new_request = await aextract(message)

        # ③ 提取器故障
        if new_request.get("error"):
            logger.warning("提取失败：%s", new_request.get("error"))
            return {"session_id": session_id, "status": "error", "version": session["version"],
                    "question": "解析服务暂时不可用，请稍后重试。"}

        # ④ 合并
        merged = merge_request(session["request"], new_request)
        missing = compute_missing(merged)
        session["request"] = merged
        session["missing"] = missing
        session["updated_at"] = time.time()

        if missing:
            return {"session_id": session_id, "status": "need_info", "version": session["version"],
                    "question": ask_for_more(missing), "missing": missing}

        # ⑤ 起任务
        session["version"] += 1
        version = session["version"]
        task_id = uuid.uuid4().hex
        tasks[task_id] = {"status": "queued", "session_id": session_id, "version": version,
                          "plan": None, "error": None, "updated_at": time.time()}
        session["planning_task"] = task_id
        asyncio.create_task(run_planning(task_id, session_id, dict(merged)))

        return {"session_id": session_id, "status": "planning", "task_id": task_id,
                "version": version, "notice": too_far_notice(merged)}
    finally:
        session["busy"] = False


@app.get("/plan/{task_id}")
async def get_plan(task_id: str):
    cleanup_expired()
    task = tasks.get(task_id)
    if task is None:
        return {"status": "failed", "error": "任务不存在或已过期", "version": 0}
    return {"status": task["status"], "plan": task["plan"],
            "error": task["error"], "version": task["version"]}


@app.get("/health")
async def health():
    return {"ok": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
