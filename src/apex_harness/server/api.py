"""FastAPI HTTP 服务：把 LLM 暴露成网页可用的流式聊天接口。

启动方式::

    uvicorn apex_harness.server.api:app --host 0.0.0.0 --port 8000

启动前需要配置 LLM（二选一）：
1. 项目根目录的 ``.env`` 文件（推荐，已加入 .gitignore）
2. 环境变量 ``APEX_LLM_API_KEY`` / ``APEX_LLM_BASE_URL`` / ``APEX_LLM_MODEL``

接口：
- ``GET  /``                返回前端聊天页面
- ``POST /chat``            提交消息，返回 ``task_id``（后台生成，与连接解耦）
- ``GET  /events/{task_id}`` 以 SSE 流式返回事件，支持 ``Last-Event-ID`` 断线补发
"""

from __future__ import annotations

import json
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from apex_harness.agent.orchestrator import Orchestrator
from apex_harness.core.kernel import Kernel
from apex_harness.llm.openai import OpenAIProvider
from apex_harness.server.tasks import TaskManager
from apex_harness.session.session import Session
from apex_harness.storage.local import LocalEventStore
from apex_harness.tools.filesystem import (
    DeleteFileTool,
    ListDirTool,
    ReadFileTool,
    WriteFileTool,
)
from apex_harness.tools.registry import ToolRegistryPlugin
from apex_harness.tools.shell import ShellExecTool
from apex_harness.tools.web import WebFetchTool

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[3]
# 前端静态目录：项目根 /web
WEB_DIR = PROJECT_ROOT / "web"


def _load_dotenv(path: Path) -> None:
    """极简 .env 加载：把 ``KEY=VALUE`` 写入环境变量（不覆盖已有值）。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


class ChatRequest(BaseModel):
    """客户端发送的聊天请求。"""

    message: str
    temperature: float = 0.7
    max_tokens: int | None = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时组装 Kernel + LLM + 工具，关闭时清理。"""
    _load_dotenv(PROJECT_ROOT / ".env")

    # 工具工作目录：可通过 APEX_WORKSPACE_DIR 指定，默认项目根目录
    workspace_dir = os.environ.get("APEX_WORKSPACE_DIR", str(PROJECT_ROOT))

    # 工具插件配置：文件系统工具限制在 workspace_dir 内
    tool_config = {
        "read_file": {"base_dir": workspace_dir},
        "write_file": {"base_dir": workspace_dir},
        "list_dir": {"base_dir": workspace_dir},
        "delete_file": {"base_dir": workspace_dir},
    }

    kernel = Kernel(config=tool_config)
    await kernel.mount_all([
        OpenAIProvider(),
        ToolRegistryPlugin(),
        ReadFileTool(),
        WriteFileTool(),
        ListDirTool(),
        DeleteFileTool(),
        ShellExecTool(),
        WebFetchTool(),
    ])
    await kernel.start()

    llm = kernel.services.require("llm")
    registry = kernel.services.require("tools")
    session = Session()

    # 作为第一条系统消息注入，每次 LLM 调用都会带上
    apex_md_path = Path(workspace_dir) / "Apex.md"
    apex_md = ""
    if apex_md_path.exists():
        apex_md = apex_md_path.read_text(encoding="utf-8")
        session.add_system(apex_md)

    orch = Orchestrator(session, llm, tools=registry.all)

    # 事件持久化 + 后台任务管理：生成与 HTTP 连接解耦，支持断线补发。
    store = LocalEventStore(PROJECT_ROOT / ".events")
    tasks = TaskManager(store)

    # 挂到 app.state，供路由使用
    app.state.kernel = kernel
    app.state.orchestrator = orch
    app.state.registry = registry
    app.state.workspace_dir = workspace_dir
    app.state.apex_md = apex_md
    app.state.tasks = tasks

    yield

    await kernel.shutdown()


app = FastAPI(title="Apex Harness", lifespan=lifespan)


class WorkspaceRequest(BaseModel):
    """设置工作目录的请求。"""

    path: str


@app.get("/api/workspace")
async def get_workspace() -> dict[str, str]:
    """返回当前工具工作目录。"""
    return {"path": app.state.workspace_dir}


@app.post("/api/workspace")
async def set_workspace(req: WorkspaceRequest) -> dict[str, str]:
    """设置工作目录：更新工具 base_dir，重新加载 Apex.md 并重建会话。"""
    new_dir = Path(req.path).resolve()
    if not new_dir.exists() or not new_dir.is_dir():
        from fastapi import HTTPException

        raise HTTPException(status_code=400, detail=f"目录不存在或不是目录: {req.path}")

    # 更新所有文件系统工具的 base_dir
    for tool in app.state.registry.all.values():
        set_dir = getattr(tool, "set_base_dir", None)
        if callable(set_dir):
            set_dir(str(new_dir))

    # 重新加载 Apex.md 并重建会话（项目说明变了，相当于新会话）
    apex_md_path = new_dir / "Apex.md"
    apex_md = ""
    if apex_md_path.exists():
        apex_md = apex_md_path.read_text(encoding="utf-8")

    llm = app.state.kernel.services.require("llm")
    registry = app.state.kernel.services.require("tools")
    session = Session()
    if apex_md:
        session.add_system(apex_md)

    app.state.workspace_dir = str(new_dir)
    app.state.apex_md = apex_md
    app.state.orchestrator = Orchestrator(session, llm, tools=registry.all)
    return {"path": str(new_dir)}


@app.get("/api/apex-md")
async def get_apex_md() -> dict[str, str]:
    """返回当前加载的 Apex.md 内容。"""
    return {"content": app.state.apex_md}


@app.post("/chat")
async def chat(request: ChatRequest) -> dict[str, str]:
    """接收一条用户消息，启动后台生成任务并返回 ``task_id``。

    生成过程在后台运行，与本次 HTTP 连接解耦：即使前端断线，后端也会继续
    生成并把事件持久化，前端可用 ``task_id`` 通过 ``/events/{task_id}``
    重连补发。
    """
    orch: Orchestrator = app.state.orchestrator
    tasks: TaskManager = app.state.tasks

    # 组装 LLM 参数，None 值不传
    params: dict[str, Any] = {
        "temperature": request.temperature,
        "top_p": request.top_p,
        "frequency_penalty": request.frequency_penalty,
        "presence_penalty": request.presence_penalty,
    }
    if request.max_tokens is not None:
        params["max_tokens"] = request.max_tokens

    task_id = uuid.uuid4().hex
    gen = orch.run_stream_events(request.message, **params)
    tasks.start(task_id, gen)
    return {"task_id": task_id}


@app.get("/events/{task_id}")
async def events(task_id: str, request: Request) -> StreamingResponse:
    """以 SSE 返回某任务的事件流，支持 ``Last-Event-ID`` 断线补发。

    事件格式为：``id: <event_id>`` + ``data: <json>``。客户端（EventSource）
    断线重连时会带上 ``Last-Event-ID``，后端据此补发遗漏事件。
    """
    tasks: TaskManager = app.state.tasks
    # EventSource 通过 Last-Event-ID 头携带已收到的最新 event_id。
    raw_last = request.headers.get("last-event-id", "0") or "0"
    try:
        after_id = int(raw_last)
    except ValueError:
        after_id = 0

    async def generate():
        async for entry in tasks.subscribe(task_id, after_id):
            yield f"id: {entry['id']}\n"
            yield f"data: {json.dumps(entry, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# 前端页面挂在根路径；必须在路由定义之后 mount，避免覆盖 /chat
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


def main() -> None:
    """本地启动入口：``python -m apex_harness.server.api``。"""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
