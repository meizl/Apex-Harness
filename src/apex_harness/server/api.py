"""FastAPI HTTP 服务：把 LLM 暴露成网页可用的流式聊天接口。

启动方式::

    uvicorn apex_harness.server.api:app --host 0.0.0.0 --port 8000

启动前需要配置 LLM（二选一）：
1. 项目根目录的 ``.env`` 文件（推荐，已加入 .gitignore）
2. 环境变量 ``APEX_LLM_API_KEY`` / ``APEX_LLM_BASE_URL`` / ``APEX_LLM_MODEL``

接口：
- ``GET  /``          返回前端聊天页面
- ``POST /chat``      发送一条消息，以流式（text/plain）返回助手回复
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from apex_harness.core.kernel import Kernel
from apex_harness.llm.openai import OpenAIProvider
from apex_harness.session.session import Session

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时组装 Kernel + LLM，关闭时清理。"""
    _load_dotenv(PROJECT_ROOT / ".env")

    kernel = Kernel()
    await kernel.mount_all([OpenAIProvider()])
    await kernel.start()

    llm = kernel.services.require("llm")
    session = Session()

    # 挂到 app.state，供路由使用
    app.state.kernel = kernel
    app.state.llm = llm
    app.state.session = session

    yield

    await kernel.shutdown()


app = FastAPI(title="Apex Harness", lifespan=lifespan)


@app.post("/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    """接收一条用户消息，流式返回助手回复（纯文本增量）。"""
    session: Session = app.state.session
    llm = app.state.llm

    # 先把用户消息写入会话，作为本次生成的上下文
    session.add_user(request.message)
    messages = session.to_llm_format()

    async def generate():
        parts: list[str] = []
        try:
            async for chunk in llm.stream(messages):
                parts.append(chunk)
                yield chunk
        except Exception as exc:  # noqa: BLE001 - 错误透传给前端
            yield f"\n\n[请求出错] {exc}"
        else:
            # 仅在成功时把完整回复写回会话，供下一轮上下文使用
            session.add_assistant("".join(parts))

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")


# 前端页面挂在根路径；必须在路由定义之后 mount，避免覆盖 /chat
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


def main() -> None:
    """本地启动入口：``python -m apex_harness.server.api``。"""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
