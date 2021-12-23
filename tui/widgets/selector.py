from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.console import RenderableType

from textual import events
from textual.widget import Widget
from textual.reactive import Reactive
from typing import List


class Selector(Widget):
    has_focus: Reactive[bool] = Reactive(False)
    mouse_over: Reactive[bool] = Reactive(False)
    cursor: Reactive[int] = Reactive(0)
    color: Reactive[str] = Reactive("blue")

    def __init__(self, title: str, elements: List) -> None:
        self.title = title
        self.elements = elements
        self.max_cursor = len(self.elements) - 1
        super().__init__(name=None)

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

    def on_key(self, event: events.Keys) -> None:
        if event.key == "up":
            self.cursor -= 1 if self.cursor > 0 else 0
        elif event.key == "down":
            self.cursor += 1 if self.cursor < self.max_cursor else 0

    def render(self) -> RenderableType:
        table = Table(
            box=box.SIMPLE,
            expand=True,
            title_style="bold",
            padding=(0, 0, 0, 0),
            show_header=False,
        )

        table.add_column(justify="center")

        for i, e in enumerate(self.elements):
            table.add_row(
                e,
                style=f"black on {self.color}" if self.cursor == i else "white",
                end_section=True,
            )

        return Panel(
            table,
            border_style="green" if self.mouse_over else "blue",
            box=box.HEAVY if self.has_focus else box.ROUNDED,
            title=self.title,
        )
