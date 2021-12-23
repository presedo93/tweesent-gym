import os

from rich import box
from rich.text import Text
from rich.tree import Tree
from rich.align import Align
from rich.panel import Panel
from rich.console import RenderableType

from textual import events
from textual.widget import Widget
from textual.reactive import Reactive


class Checkpoints(Widget):
    has_focus: Reactive[bool] = Reactive(False)
    mouse_over: Reactive[bool] = Reactive(False)
    cursor: Reactive[int] = Reactive(-1)
    color: Reactive[str] = Reactive("blue")
    max_cursor = 0
    files = []

    def __init__(self, folder: str) -> None:
        self.folder = folder
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
            self.cursor -= 1 if self.cursor > -1 else 0
        elif event.key == "down":
            self.cursor += 1 if self.cursor < self.max_cursor else 0

    def render(self) -> RenderableType:
        if os.path.exists("tb_logs"):
            return self.checks_logs()
        else:
            return self.no_logs()

    def checks_logs(self) -> RenderableType:
        tree = Tree(
            f"models & datasets", guide_style="green" if self.mouse_over else "blue"
        )

        tree.add(
            "no checkpoint",
            style=f"black on {self.color}" if self.cursor == -1 else "white",
        )
        for (i, model) in enumerate(os.listdir("tb_logs")):
            model_tree = tree.add(model)
            for (j, dataset) in enumerate(os.listdir(f"tb_logs/{model}")):
                self.max_cursor = i + j
                self.files.append(f"{model}/{dataset}")
                model_tree.add(
                    dataset,
                    style=f"black on {self.color}" if self.cursor == i + j else "white",
                )

        return Panel(
            tree,
            border_style="green" if self.mouse_over else "blue",
            box=box.HEAVY if self.has_focus else box.ROUNDED,
            title="checkpoints",
        )

    def no_logs(self) -> RenderableType:
        return Panel(
            Align.center(Text("There are no logs available"), vertical="middle"),
            border_style="green" if self.mouse_over else "blue",
            box=box.HEAVY if self.has_focus else box.ROUNDED,
            title="devices",
        )
