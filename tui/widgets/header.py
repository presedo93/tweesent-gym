from datetime import datetime as dt

from rich.panel import Panel
from rich.table import Table

from textual.widget import Widget


class Header(Widget):
    def on_mount(self):
        self.set_interval(1, self.refresh)

    def render(self):
        grid = Table.grid(expand=True)
        grid.add_column(justify="center", ratio=1)
        grid.add_column(justify="right", style="white")
        grid.add_row("[b blue]TweeSent Gym![/]", dt.now().ctime())

        return Panel(grid, style="blue")
