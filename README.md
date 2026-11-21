# 旅小智 · AI 旅行规划智能体

> 一句话：说清"去哪儿、玩几天"，一套基于 **LangGraph** 的多智能体系统在 2–3 分钟内产出一份带天气、预算、住宿、交通、美食的七章旅行计划。

---

## 架构

```
用户一句话
   │
   ▼
POST /chat ──► 请求理解层（LLM 结构化抽取）──► 缺字段？──► 生成追问，等补充
   │                                        │
   │ 字段齐了                                ▼
   │                                  TravelRequest
   ▼
┌──────────────────────────────────────────────────────────┐
│                     LangGraph 状态图                      │
│                                                          │
│                     ┌──────────────┐                     │
│        ┌───────────►│  coordinator │◄──────────┐         │
│        │            └──────┬───────┘           │         │
│        │    条件边（看当前 state 缺什么信息）      │         │
│        │   ┌───────┬───────┼───────┬───────┐   │         │
│        ▼   ▼       ▼       ▼       ▼       ▼   │         │
│  travel_advisor  weather_analyst  budget_optimizer      │
│                  local_expert     itinerary_planner     │
│        │            │           │           │           │
│        └────────────┴───────────┴───────────┘           │
│                   每位专家：bind_tools 原生工具调用 / prefetch│
│                             │                            │
│                    信息够了 ▼                            │
│                        plan_writer ──► final_plan（七章） │
└──────────────────────────────────────────────────────────┘
   │
   ▼
Streamlit 前端（对话式）── 进度轮询 ── 计划抬头 + Markdown + 下载
```

**7 个节点**：`coordinator`（唯一入口 + 汇合点）、5 位专家（旅行顾问 / 天气分析师 / 预算优化师 / 当地专家 / 行程规划师）、`plan_writer`（汇总）。

**两处循环**：专家 → 工具 → 专家（ReAct），专家 → coordinator → 派下一个专家。

## 核心能力

- **自然语言进，计划出**：`今日 → 目的地/天数/预算/同行人/偏好` 结构化抽取，缺字段会追问（不猜）
- **多轮改需求**：同一会话说「改成5天」，在已收集的需求上合并，而不是从头再来
- **原生工具调用**：`bind_tools` 的 function calling，非文本协议；全链路 async
- **手写 MCP 天气服务器**：FastMCP + stdio 传输，独立客户端联调验证，再接入智能体
- **并发保护 + 长任务可观测**：单图串行锁、任务 ID + 轮询、`queued/running/done/failed` 状态
- **可演示的前端**：对话式界面、计划抬头显示关键假设（人数缺省、预算未指定）、Markdown 下载

## 技术栈

| 层 | 选型 |
|---|---|
| 编排 | LangGraph 0.3（StateGraph，增量 patch，copy-on-write） |
| 模型 | DeepSeek（OpenAI 兼容接口，`langchain-openai`） |
| 工具协议 | MCP（FastMCP 服务端 + `langchain-mcp-adapters`） |
| 服务 | FastAPI + 内存会话（TTL / 容量上限） |
| 前端 | Streamlit（`st.chat_input` 对话式 + 轮询） |
| 部署 | Docker / Docker Compose |

## 快速开始

### 方式一：Docker（推荐）

```bash
cp .env.example .env      # 填入自己的 key
docker compose up -d --build
# 前端 http://localhost:8501    后端 http://localhost:8000/health
```

### 方式二：本地运行

```bash
pip install -r requirements.txt

# 终端 A：后端
python -m uvicorn api.api_server:app --host 127.0.0.1 --port 8000
# 终端 B：前端
python -m streamlit run front/streamlit_app.py --server.port 8501
```

> Windows 上中文日志建议先 `set PYTHONIOENCODING=utf-8`（或 `$env:PYTHONIOENCODING='utf-8'`）。

## 配置

| 变量 | 说明 |
|---|---|
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL` | LLM（兼容 OpenAI 协议的服务都可） |
| `BOCHA_API_KEY` | 博查搜索，用于景点/酒店/餐厅/贴士 |
| `QWEATHER_API_KEY` / `QWEATHER_API_BASE` | 和风天气，需填专属 API Host |
| `WEATHER_USE_MCP` | `1` 走 MCP 服务器（默认），`0` 走 HTTP 兜底 |


## 目录结构

```
agents/          多智能体图（langgraph_agents.py）、请求理解层（request_extractor.py）
api/             FastAPI 服务（会话、任务、轮询）
config/          环境与 LLM 配置
tools/           搜索/天气等工具；weather_mcp_server.py 是独立的 MCP 服务器
front/           Streamlit 前端
```

