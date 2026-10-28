"""Quick test script for the plugin kernel.

Run with:  python -m test_kernel

Demonstrates:
- Plugin A provides a service, Plugin B requires it
- Event communication between plugins
- Auto-cleanup on shutdown
"""

import asyncio

from apex_harness.core.events import Event
from apex_harness.core.kernel import Kernel
from apex_harness.core.plugin import Plugin


# ---------------------------------------------------------------------------
# 1. Define a custom event
# ---------------------------------------------------------------------------

class GreetEvent(Event):
    name = "greet.sent"


# ---------------------------------------------------------------------------
# 2. Plugin A: provides the "greeter" service and emits an event
# ---------------------------------------------------------------------------

class GreeterPlugin(Plugin):
    name = "greeter"
    provides = ["greeter"]

    async def setup(self):
        print(f"[{self.name}] setup: 注册 greeter 服务")
        self.ctx.register_service("greeter", self)

    async def start(self):
        print(f"[{self.name}] start: 发送 greet.sent 事件")
        await self.ctx.emit(GreetEvent(text="你好，世界！"))

    def greet(self, name: str) -> str:
        return f"你好, {name}!"


# ---------------------------------------------------------------------------
# 3. Plugin B: requires the "greeter" service and listens for the event
# ---------------------------------------------------------------------------

class UserPlugin(Plugin):
    name = "user"
    requires = ["greeter"]

    async def setup(self):
        print(f"[{self.name}] setup: 监听 greet.sent 事件")
        self.ctx.on("greet.sent", self.on_greet)

    async def start(self):
        greeter = self.ctx.require_service("greeter")
        print(f"[{self.name}] start: 调用 greeter → {greeter.greet('小明')}")

    async def on_greet(self, event: GreetEvent):
        print(f"[{self.name}] 收到事件: {event.payload['text']}")


# ---------------------------------------------------------------------------
# 4. Run the kernel
# ---------------------------------------------------------------------------

async def main():
    print("=== 创建 Kernel ===")
    kernel = Kernel()

    print("\n=== mount_all (顺序随便传，内核自动排序) ===")
    await kernel.mount_all([UserPlugin(), GreeterPlugin()])

    print(f"\n已挂载的插件: {list(kernel.plugins.keys())}")
    print(f"已注册的服务: {kernel.services.names()}")

    print("\n=== start ===")
    await kernel.start()

    print("\n=== shutdown (验证自动清理) ===")
    await kernel.shutdown()

    print(f"\n清理后服务表: {kernel.services.names()} (应为空)")
    print("\n=== 完成 ===")


if __name__ == "__main__":
    asyncio.run(main())
