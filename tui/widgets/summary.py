from rich import box
from rich.align import Align
from rich.panel import Panel
from rich.console import RenderableType

from textual import events
from textual.widget import Widget
from textual.reactive import Reactive


class Summary(Widget):
    has_focus: Reactive[bool] = Reactive(False)
    mouse_over: Reactive[bool] = Reactive(False)
    color: Reactive[str] = Reactive("blue")

    async def on_focus(self, event: events.Focus) -> None:
        self.has_focus = True

    async def on_blur(self, event: events.Blur) -> None:
        self.has_focus = False

    async def on_enter(self, event: events.Enter) -> None:
        self.mouse_over = True
        self.color = "green"

    async def on_leave(self, event: events.Leave) -> None:
        self.mouse_over = False
        self.color = "blue"

    def render(self) -> RenderableType:
        text = f"[{'green' if self.mouse_over else 'blue'}]Once a task is completed, summary will be shown here[/]"
        return Panel(
            Align.center(text, vertical="middle"),
            border_style="green" if self.mouse_over else "blue",
            box=box.HEAVY if self.has_focus else box.ROUNDED,
            title="runner",
        )
