from pathlib import Path
import os, sys
from typing import Dict
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent   # 项目根
PYTHON_EXE = sys.executable


def build_client(python_exe: str = PYTHON_EXE,
                 project_root: Path = PROJECT_ROOT) -> MultiServerMCPClient:
    return MultiServerMCPClient({
        "weather": {
            "transport": "stdio",
            "command": python_exe,
            "args": ["-m", "tools.weather_mcp_server"],
            "cwd": str(project_root),
            "env": {**os.environ, "PYTHONIOENCODING": "utf-8"},
            "encoding": "utf-8",
        }
    })


async def fetch_weather_tools(client: MultiServerMCPClient | None = None) -> Dict[str, BaseTool]:
    client = client or build_client()
    tools = await client.get_tools()
    return {tool.name: tool for tool in tools}