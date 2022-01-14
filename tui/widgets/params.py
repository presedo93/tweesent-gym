from rich import box
from rich.align import Align
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.padding import Padding
from rich.console import RenderableType

from textual import events
from textual.widget import Widget
from textual.reactive import Reactive

from typing import Dict
from tools.utils import open_conf

from tui.messages import EnterCommand


class Params(Widget):
    has_focus: Reactive[bool] = Reactive(False)
    mouse_over: Reactive[bool] = Reactive(False)
    enabled: Reactive[bool] = Reactive(False)
    text: Reactive[str] = Reactive("")
    conf: Reactive[Dict] = Reactive({})

    def on_mount(self) -> None:
        self.conf = open_conf("conf/conf.json")

    async def on_focus(self, event: events.Focus) -> None:
        self.has_focus = True

    async def on_blur(self, event: events.Blur) -> None:
        self.has_focus = False

    async def on_enter(self, event: events.Enter) -> None:
        self.mouse_over = True

    async def on_leave(self, event: events.Leave) -> None:
        self.mouse_over = False

    async def handle_enter_command(self, message: EnterCommand) -> None:
        self.conf["train"]["batch_size"] += 1
        self.refresh()

    def render(self) -> RenderableType:
        train_table = self.create_table(self.conf["train"], "Train")
        test_table = self.create_table(self.conf["test"], "Test")
        pred_table = self.create_table(self.conf["predict"], "Predict")

        layout = Layout()
        layout.split_column(
            Layout(name="upper"), Layout(name="middle"), Layout(name="bottom")
        )

        layout["upper"].update(Padding(Align.center(train_table), 2))
        layout["middle"].update(Padding(Align.center(test_table), 4))
        layout["bottom"].update(Padding(Align.center(pred_table), 4))

        return Panel(
            layout,
            border_style="green" if self.mouse_over else "blue",
            box=box.HEAVY if self.has_focus else box.ROUNDED,
            title="parameters",
        )

    def create_table(self, conf: Dict, stage: str) -> Table:
        table = Table(
            title=f"{stage} Parameters",
            show_header=False,
            box=box.ROUNDED,
            width=100,
            border_style="green" if self.mouse_over else "blue",
            show_lines=True,
            title_style="bold",
        )
        table.add_column(justify="center")
        table.add_column(justify="center")
        table.add_column(justify="center")
        table.add_column(justify="center")

        params = [(k, v) for k, v in conf.items()]
        for i, _ in enumerate(params):
            if i % 2 != 0:
                continue
            try:
                param1, val1 = str(params[i][0]), str(params[i][1])
            except IndexError:
                param1 = val1 = "-" * 10

            try:
                param2, val2 = str(params[i + 1][0]), str(params[i + 1][1])
            except IndexError:
                param2 = val2 = "-" * 10
            table.add_row(param1, val1, param2, val2)

        return table
