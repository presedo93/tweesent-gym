from rich import box
from rich.panel import Panel

from textual import events
from textual.widget import Widget
from textual.message import Message
from textual.reactive import Reactive


class EnterPressed(Message, bubble=True):
    pass


class Command(Widget):
    has_focus: Reactive[bool] = Reactive(False)
    mouse_over: Reactive[bool] = Reactive(False)
    text: Reactive[str] = Reactive("")

    async def on_focus(self, event: events.Focus) -> None:
        self.has_focus = True

    async def on_blur(self, event: events.Blur) -> None:
        self.has_focus = False

    async def on_enter(self, event: events.Enter) -> None:
        self.mouse_over = True

    async def on_leave(self, event: events.Leave) -> None:
        self.mouse_over = False

    async def on_key(self, event) -> None:
        if event.key == "ctrl+h":
            self.text = self.text[:-1]
        if event.key == "enter":
            await self.post_message(EnterPressed(self))
            await self.emit(EnterPressed(self))
        else:
            self.text += event.key

    def render(self) -> Panel:
        return Panel(
            f"[b white]Command:[not b] {self.text}[/]",
            box=box.HEAVY if self.has_focus else box.ROUNDED,
            border_style="green" if self.mouse_over else "blue",
        )
