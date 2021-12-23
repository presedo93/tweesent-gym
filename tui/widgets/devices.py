import re
import subprocess

from rich import box
from rich.align import Align
from rich.panel import Panel
from rich.table import Table
from rich.console import RenderableType

from textual import events
from textual.widget import Widget
from textual.reactive import Reactive


from typing import Dict


class DeviceStats(Widget):
    has_focus: Reactive[bool] = Reactive(False)
    mouse_over: Reactive[bool] = Reactive(False)
    enabled: Reactive[bool] = Reactive(False)
    stats: Dict = {}

    def on_mount(self):
        self.timer_stats = self.set_interval(5, self.refresh)

    async def on_focus(self, event: events.Focus) -> None:
        self.has_focus = True

    async def on_blur(self, event: events.Blur) -> None:
        self.has_focus = False

    async def on_enter(self, event: events.Enter) -> None:
        self.mouse_over = True

    async def on_leave(self, event: events.Leave) -> None:
        self.mouse_over = False

    async def on_click(self, event) -> None:
        self.enabled = not self.enabled

    def render(self) -> RenderableType:
        if self.enabled:
            self.timer_stats.resume()
            return self.panel_gpu_stats()
        else:
            self.timer_stats.pause()
            return self.panel_to_enable()

    def panel_to_enable(self) -> RenderableType:
        grid = Table(
            style=f"{'green' if self.mouse_over else 'blue'}",
            show_header=False,
            box=box.ROUNDED,
        )
        grid.add_column(justify="center")
        grid.add_row("Enable GPU Stats")

        return Panel(
            Align.center(grid, vertical="middle"),
            border_style="green" if self.mouse_over else "blue",
            box=box.HEAVY if self.has_focus else box.ROUNDED,
            title="devices",
        )

    def panel_gpu_stats(self) -> RenderableType:
        self.fetch_gpu()
        table = Table(
            title=f"[black on {'green' if self.mouse_over else 'blue'}]GPU stats[/]",
            box=None,
            expand=True,
            title_style="bold",
            padding=(1, 1, 0, 1),
        )
        table.add_column("Model", justify="left", ratio=1)
        table.add_column(f"{self.stats['name']}", justify="left", ratio=1)

        table.add_row(
            "Memory Usage", f"{self.stats['mem_used']}/{self.stats['mem_total']}"
        )
        table.add_row(
            "Power Consumed:", f"{self.stats['power_draw']}/{self.stats['power_limit']}"
        )

        return Panel(
            table,
            border_style="green" if self.mouse_over else "blue",
            box=box.HEAVY if self.has_focus else box.ROUNDED,
            title="devices",
        )

    def fetch_gpu(self) -> Dict:
        cmd = subprocess.run(["nvidia-smi", "-q"], stdout=subprocess.PIPE)
        stdout = cmd.stdout.decode("utf-8")

        self.stats["name"] = re.findall("Product Name\s*:\s*(.*)", stdout)[0]
        self.stats["mem_used"] = re.findall(
            "FB Memory Usage\s*.*\s*Used\s*:\s*(.*)", stdout
        )[0]
        self.stats["mem_total"] = re.findall(
            "FB Memory Usage\s*.*\s*Total\s*:\s*(.*)", stdout
        )[0]
        self.stats["power_draw"] = re.findall("Power Draw\s*:\s*(.*)", stdout)[0]
        self.stats["power_limit"] = re.findall("\s\sPower Limit\s*:\s*(.*)", stdout)[0]
