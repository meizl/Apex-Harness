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

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import SystemMessage, message_to_dict
from pydantic import BaseModel, Field

from apex_harness.agent.orchestrator import Orchestrator
from apex_harness.core.kernel import Kernel
from apex_harness.llm.openai import OpenAIProvider
from apex_harness.server.tasks import TaskManager
from apex_harness.session.checkpoint import load_checkpoint, save_checkpoint
from apex_harness.session.session import Session
from apex_harness.skills import (
    USER_SKILLS_DIR,
    SkillMetadata,
    SkillRegistryPlugin,
    load_builtin_skills,
    load_skills_from_dir,
)
from apex_harness.skills.parser import parse_skill_md
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
# 会话 checkpoint 目录：按 workspace hash 隔离存储
SESSIONS_DIR = PROJECT_ROOT / "sessions"


def build_system_prompt(apex_md: str, skills_metadata: list[SkillMetadata]) -> str:
    """构建系统提示：Apex.md + 可用技能摘要列表。

    渐进式披露：只注入技能的 name + description，不注入完整 instructions。
    完整指令在 LLM 调用该技能时由 SkillToolAdapter 返回。
    """
    parts: list[str] = []
    if apex_md:
        parts.append(apex_md)

    if skills_metadata:
        skill_lines = ["## 可用技能", "", "你可以使用以下技能（通过函数调用激活）："]
        for m in skills_metadata:
            skill_lines.append(f"- `{m.name}`：{m.description}")
        skill_lines.append("")
        skill_lines.append("当用户询问你有哪些技能或能力时，请列出以上技能。")
        parts.append("\n".join(skill_lines))

    return "\n\n".join(parts)


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
    max_tokens: Optional[int] = None
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

    # 收集技能元数据：内置（SKILL.md）+ 外部目录（APEX_SKILLS_DIR，可选）
    # 渐进式披露：启动时只注册 name/description，LLM 调用时才注入完整 instructions。
    skills_metadata = load_builtin_skills()
    external_skills_dir = os.environ.get("APEX_SKILLS_DIR")
    if external_skills_dir:
        skills_metadata.extend(load_skills_from_dir(external_skills_dir))

    await kernel.mount_all([
        OpenAIProvider(),
        ToolRegistryPlugin(),
        SkillRegistryPlugin(skills_metadata),
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

    # 作为第一条系统消息注入，每次 LLM 调用都会带上
    apex_md_path = Path(workspace_dir) / "Apex.md"
    apex_md = ""
    if apex_md_path.exists():
        apex_md = apex_md_path.read_text(encoding="utf-8")

    # 系统提示 = Apex.md + 可用技能摘要（渐进式披露，不含完整 instructions）
    system_prompt = build_system_prompt(apex_md, skills_metadata)

    # 从 checkpoint 恢复会话（按 workspace hash 隔离）；不存在则新建
    session = load_checkpoint(SESSIONS_DIR, workspace_dir)
    if session is None:
        session = Session()
        if system_prompt:
            session.add_system(system_prompt)
    else:
        # 恢复后同步最新系统提示（Apex.md 或技能列表可能变了）
        if session.messages and session.messages[0].type == "system":
            if system_prompt:
                session.messages[0] = SystemMessage(content=system_prompt)
            else:
                session.messages.pop(0)
        elif system_prompt:
            session.messages.insert(0, SystemMessage(content=system_prompt))

    orch = Orchestrator(session, llm, tools=registry.all)

    # 事件持久化 + 后台任务管理：生成与 HTTP 连接解耦，支持断线补发。
    store = LocalEventStore(PROJECT_ROOT / ".events")
    tasks = TaskManager(store)

    # 技能进化（右循环）：待审批队列 + SkillBank + Worker
    from apex_harness.skills.evolution import (
        PendingSkillStore,
        SkillBank,
        SkillEvolutionWorker,
    )

    sessions_dir = PROJECT_ROOT / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    pending_store = PendingSkillStore(sessions_dir / "pending_skills.jsonl")
    skill_bank = SkillBank(sessions_dir / "skillbank.jsonl")
    evolution_worker = SkillEvolutionWorker(llm, skill_bank, pending_store)

    # 挂到 app.state，供路由使用
    app.state.kernel = kernel
    app.state.orchestrator = orch
    app.state.registry = registry
    app.state.skills_metadata = skills_metadata
    app.state.workspace_dir = workspace_dir
    app.state.apex_md = apex_md
    app.state.tasks = tasks
    app.state.pending_store = pending_store
    app.state.skill_bank = skill_bank
    app.state.evolution_worker = evolution_worker

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

    # 先保存当前 workspace 的会话
    try:
        save_checkpoint(
            app.state.orchestrator.session, SESSIONS_DIR, str(app.state.workspace_dir)
        )
    except Exception:
        pass

    # 更新所有文件系统工具的 base_dir
    for tool in app.state.registry.all.values():
        set_dir = getattr(tool, "set_base_dir", None)
        if callable(set_dir):
            set_dir(str(new_dir))

    # 重新加载 Apex.md 并从 checkpoint 恢复新 workspace 的会话
    apex_md_path = new_dir / "Apex.md"
    apex_md = ""
    if apex_md_path.exists():
        apex_md = apex_md_path.read_text(encoding="utf-8")

    llm = app.state.kernel.services.require("llm")
    registry = app.state.kernel.services.require("tools")

    # 系统提示 = Apex.md + 可用技能摘要
    skills_metadata = app.state.skills_metadata
    system_prompt = build_system_prompt(apex_md, skills_metadata)

    session = load_checkpoint(SESSIONS_DIR, str(new_dir))
    if session is None:
        session = Session()
        if system_prompt:
            session.add_system(system_prompt)
    else:
        if session.messages and session.messages[0].type == "system":
            if system_prompt:
                session.messages[0] = SystemMessage(content=system_prompt)
            else:
                session.messages.pop(0)
        elif system_prompt:
            session.messages.insert(0, SystemMessage(content=system_prompt))

    app.state.workspace_dir = str(new_dir)
    app.state.apex_md = apex_md
    app.state.orchestrator = Orchestrator(session, llm, tools=registry.all)
    return {"path": str(new_dir)}


@app.get("/api/apex-md")
async def get_apex_md() -> dict[str, str]:
    """返回当前加载的 Apex.md 内容。"""
    return {"content": app.state.apex_md}


@app.get("/api/messages")
async def get_messages() -> list[dict[str, Any]]:
    """返回当前会话的所有消息，供前端刷新后恢复展示。"""
    session: Session = app.state.orchestrator.session
    return [message_to_dict(m) for m in session.messages]


@app.get("/api/skills")
async def get_skills() -> list[dict[str, Any]]:
    """返回当前可用的技能列表（name + description + 是否内置）。"""
    skills: list[SkillMetadata] = app.state.skills_metadata
    result = []
    for m in skills:
        is_builtin = not _is_user_skill(m.name)
        result.append({"name": m.name, "description": m.description, "is_builtin": is_builtin})
    return result


# ---- 待审批技能（右循环产出，需用户审批后入库）---- #

@app.get("/api/skills/pending")
async def list_pending_skills() -> list[dict[str, Any]]:
    """返回待审批的候选技能列表。"""
    from apex_harness.skills.evolution.models import PendingSkill

    pending_store = app.state.pending_store
    items: list[PendingSkill] = pending_store.list_pending()
    return [
        {
            "pending_id": p.pending_id,
            "name": p.candidate.name,
            "description": p.candidate.description,
            "instruction": p.candidate.instruction,
            "trigger_conditions": p.candidate.trigger_conditions,
            "tools": p.candidate.tools,
            "examples": p.candidate.examples,
            "confidence_score": p.candidate.confidence_score,
            "source_trace_id": p.candidate.source_trace_id,
            "action": p.decision.action.value,
            "reason": p.decision.reason,
            "created_at": p.created_at.isoformat(),
        }
        for p in items
    ]


class PendingUpdateRequest(BaseModel):
    """用户修改待审批候选的请求。"""
    name: str = ""
    description: str = ""
    instruction: str = ""
    trigger_conditions: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)


@app.put("/api/skills/pending/{pending_id}")
async def update_pending_skill(
    pending_id: str, req: PendingUpdateRequest
) -> dict[str, Any]:
    """用户修改待审批候选的内容。"""
    from fastapi import HTTPException
    from apex_harness.skills.evolution.models import CandidateSkill

    pending_store = app.state.pending_store
    pending = pending_store.get(pending_id)
    if pending is None:
        raise HTTPException(status_code=404, detail="待审批技能不存在")

    # 基于原候选构建修改后的版本
    old = pending.candidate
    updated = CandidateSkill(
        name=req.name or old.name,
        description=req.description if req.description is not None else old.description,
        instruction=req.instruction if req.instruction is not None else old.instruction,
        trigger_conditions=req.trigger_conditions or old.trigger_conditions,
        tools=req.tools or old.tools,
        examples=req.examples or old.examples,
        source_trace_id=old.source_trace_id,
        confidence_score=old.confidence_score,
    )
    pending_store.update_candidate(pending_id, updated)
    return {"message": "已更新", "pending_id": pending_id}


@app.post("/api/skills/pending/{pending_id}/approve")
async def approve_pending_skill(pending_id: str) -> dict[str, Any]:
    """审批通过：将候选写入 SkillBank（并热加载为可用技能）。"""
    from fastapi import HTTPException

    worker = app.state.evolution_worker
    result = await worker.approve(pending_id)
    if result is None:
        raise HTTPException(status_code=404, detail="待审批技能不存在")

    entry = result.skill_entry
    if entry is None:
        raise HTTPException(status_code=500, detail="写入 SkillBank 失败")

    # 热加载为可用技能（写入 user 目录 + 注册）
    from apex_harness.skills.parser import parse_skill_md
    from apex_harness.skills.base import SkillMetadata
    from apex_harness.skills.adapter import SkillToolAdapter

    content = (
        f"---\nname: {entry.name}\n"
        f"description: {entry.description}\n"
        f"---\n\n{entry.instruction}\n"
    )
    # 校验格式
    try:
        parsed = parse_skill_md(content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"格式错误：{e}")

    # 保存到用户技能目录
    skill_dir = USER_SKILLS_DIR / entry.name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")

    # 注册到技能注册表 + 工具注册表
    metadata = SkillMetadata(
        name=parsed.name,
        description=parsed.description,
        instructions=parsed.instructions,
    )
    skill_registry = app.state.kernel.services.require("skills")
    tool_registry = app.state.kernel.services.require("tools")
    skill_registry.register(metadata)
    tool_registry.register(SkillToolAdapter(metadata))

    existing_names = {m.name for m in app.state.skills_metadata}
    if parsed.name not in existing_names:
        app.state.skills_metadata.append(metadata)

    orch: Orchestrator = app.state.orchestrator
    orch.react.tools[metadata.name] = SkillToolAdapter(metadata)

    return {
        "message": f"技能 {entry.name} 已审批通过并加载",
        "name": entry.name,
        "version": entry.version,
    }


@app.post("/api/skills/pending/{pending_id}/reject")
async def reject_pending_skill(pending_id: str) -> dict[str, Any]:
    """审批拒绝：丢弃候选。"""
    from fastapi import HTTPException

    worker = app.state.evolution_worker
    ok = await worker.reject(pending_id)
    if not ok:
        raise HTTPException(status_code=404, detail="待审批技能不存在")
    return {"message": "已拒绝"}


@app.get("/api/skills/{skill_name}")
async def get_skill_detail(skill_name: str) -> dict[str, str]:
    """返回指定技能的完整信息（含 instructions）。"""
    skill_registry = app.state.kernel.services.require("skills")
    meta = skill_registry.get(skill_name)
    if meta is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"技能 {skill_name} 不存在")
    return {
        "name": meta.name,
        "description": meta.description,
        "instructions": meta.instructions,
    }


class SkillCreateRequest(BaseModel):
    """新增技能请求：可传完整 content，或分字段传入。"""
    name: str = ""
    description: str = ""
    instructions: str = ""
    content: str = ""  # 完整 SKILL.md 文本，优先级高于分字段


@app.post("/api/skills")
async def create_skill(req: SkillCreateRequest) -> dict[str, Any]:
    """新增一个技能：校验格式 → 保存到用户技能目录 → 热加载到注册表。"""
    # 1. 组装 SKILL.md 内容
    if req.content.strip():
        content = req.content
    else:
        if not req.name.strip():
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail="技能名称 name 不能为空")
        content = f"---\nname: {req.name}\ndescription: {req.description}\n---\n\n{req.instructions}\n"

    # 2. 校验格式
    try:
        parsed = parse_skill_md(content)
    except ValueError as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=f"格式错误：{e}")

    if not parsed.name.strip():
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="格式错误：frontmatter 缺少必填字段 name")

    # 3. 保存到用户技能目录
    skill_dir = USER_SKILLS_DIR / parsed.name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")

    # 4. 热加载：注册到 SkillRegistry + ToolRegistry，更新 Orchestrator 的 tools
    metadata = SkillMetadata(
        name=parsed.name,
        description=parsed.description,
        instructions=parsed.instructions,
    )

    from apex_harness.skills.adapter import SkillToolAdapter

    skill_registry = app.state.kernel.services.require("skills")
    tool_registry = app.state.kernel.services.require("tools")

    # 注册到技能注册表
    skill_registry.register(metadata)
    # 注册为工具（LLM 可调用）
    tool_registry.register(SkillToolAdapter(metadata))

    # 更新 app.state 的技能列表
    existing_names = {m.name for m in app.state.skills_metadata}
    if parsed.name not in existing_names:
        app.state.skills_metadata.append(metadata)

    # 更新 Orchestrator 的 tools 字典，让新技能立即生效
    orch: Orchestrator = app.state.orchestrator
    orch.react.tools[metadata.name] = SkillToolAdapter(metadata)

    return {
        "name": parsed.name,
        "description": parsed.description,
        "message": f"技能 {parsed.name} 已创建并加载",
    }


def _get_user_skill_path(skill_name: str) -> Path:
    """返回用户技能目录下指定技能的 SKILL.md 路径。"""
    return USER_SKILLS_DIR / skill_name / "SKILL.md"


def _is_user_skill(skill_name: str) -> bool:
    """判断技能是否为用户创建（文件在 user/ 目录下）。"""
    return _get_user_skill_path(skill_name).exists()


@app.delete("/api/skills/{skill_name}")
async def delete_skill(skill_name: str) -> dict[str, Any]:
    """删除用户技能：移除文件 + 从注册表卸载。

    内置技能（builtins/）不允许删除。
    """
    from fastapi import HTTPException
    from apex_harness.skills.adapter import SkillToolAdapter

    if not _is_user_skill(skill_name):
        raise HTTPException(status_code=403, detail=f"技能 {skill_name} 是内置技能，不允许删除")

    skill_path = _get_user_skill_path(skill_name)
    skill_dir = skill_path.parent

    # 1. 删除文件
    import shutil
    shutil.rmtree(skill_dir, ignore_errors=True)

    # 2. 从注册表卸载
    skill_registry = app.state.kernel.services.require("skills")
    tool_registry = app.state.kernel.services.require("tools")
    skill_registry.remove(skill_name)
    tool_registry.remove(skill_name)

    # 3. 从 app.state 技能列表移除
    app.state.skills_metadata = [m for m in app.state.skills_metadata if m.name != skill_name]

    # 4. 从 Orchestrator 的 tools 移除
    orch: Orchestrator = app.state.orchestrator
    orch.react.tools.pop(skill_name, None)

    return {"message": f"技能 {skill_name} 已删除"}


@app.put("/api/skills/{skill_name}")
async def update_skill(skill_name: str, req: SkillCreateRequest) -> dict[str, Any]:
    """更新用户技能：校验格式 → 覆盖文件 → 热更新注册表。"""
    from fastapi import HTTPException
    from apex_harness.skills.adapter import SkillToolAdapter

    if not _is_user_skill(skill_name):
        raise HTTPException(status_code=403, detail=f"技能 {skill_name} 是内置技能，不允许修改")

    # 1. 组装内容（name 锁定为路径中的 skill_name，防止改名）
    if req.content.strip():
        content = req.content
    else:
        content = f"---\nname: {skill_name}\ndescription: {req.description}\n---\n\n{req.instructions}\n"

    # 2. 校验格式
    try:
        parsed = parse_skill_md(content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"格式错误：{e}")

    if parsed.name != skill_name:
        raise HTTPException(status_code=400, detail="name 字段与路径不一致，不允许改名")

    # 3. 覆盖文件
    skill_path = _get_user_skill_path(skill_name)
    skill_path.write_text(content, encoding="utf-8")

    # 4. 热更新注册表
    metadata = SkillMetadata(
        name=parsed.name,
        description=parsed.description,
        instructions=parsed.instructions,
    )

    skill_registry = app.state.kernel.services.require("skills")
    tool_registry = app.state.kernel.services.require("tools")

    skill_registry.register(metadata)  # register 会覆盖同名
    tool_registry.remove(skill_name)
    tool_registry.register(SkillToolAdapter(metadata))

    # 更新 app.state
    app.state.skills_metadata = [m for m in app.state.skills_metadata if m.name != skill_name]
    app.state.skills_metadata.append(metadata)

    # 更新 Orchestrator tools
    orch: Orchestrator = app.state.orchestrator
    orch.react.tools[skill_name] = SkillToolAdapter(metadata)

    return {
        "name": parsed.name,
        "description": parsed.description,
        "message": f"技能 {parsed.name} 已更新",
    }


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
    task = tasks.start(task_id, gen)

    # 任务完成后异步保存 checkpoint（不阻塞返回 task_id）
    async def _save_after_done() -> None:
        try:
            await task
        except Exception:
            pass
        try:
            save_checkpoint(orch.session, SESSIONS_DIR, str(app.state.workspace_dir))
        except Exception:
            pass

    asyncio.create_task(_save_after_done())

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


@app.post("/tasks/{task_id}/cancel")
async def cancel_task(task_id: str) -> dict[str, Any]:
    """取消指定后台生成任务。

    成功后任务会被中断，并产生一条 ``cancelled`` 终态事件，供订阅者与回放使用。
    """
    tasks: TaskManager = app.state.tasks
    cancelled = await tasks.cancel(task_id)
    return {"task_id": task_id, "cancelled": cancelled}


# 前端页面挂在根路径；必须在路由定义之后 mount，避免覆盖 /chat
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


def main() -> None:
    """本地启动入口：``python -m apex_harness.server.api``。"""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
