from logging import PlaceHolder
from textual.widgets import Placeholder
from textual.views import GridView

from tui.widgets.gpusts import GPUStats
from tui.widgets.params import Params


class DashGrid(GridView):
    def on_mount(self) -> None:
        """Make a simple grid arrangement."""

        self.grid.add_column(fraction=1, name="left")
        self.grid.add_column(fraction=1, name="leftcent")
        self.grid.add_column(fraction=2, name="rightcent")
        self.grid.add_column(fraction=2, name="right")

        self.grid.add_row(fraction=1, name="top")
        self.grid.add_row(fraction=1, name="midtop")
        self.grid.add_row(fraction=1, name="middle")
        self.grid.add_row(fraction=1, name="midbot")
        self.grid.add_row(fraction=1, name="bottom")

        self.grid.add_areas(
            area1="left-start|leftcent-end,top",
            area2="rightcent-start|right-end,top-start|midbot-end",
            area3="left-start|right-end,bottom",
            area4="left-start|leftcent-end,midtop",
            area5="left,middle",
            area6="leftcent,middle",
            area7="left,midbot",
            area8="leftcent, midbot",
        )

        self.grid.place(
            area1=GPUStats(name="area1"),
            area2=Params(name="area2"),
            area3=Placeholder(name="area3"),
            area4=Placeholder(name="area4"),
            area5=Placeholder(name="area5"),
            area6=Placeholder(name="area6"),
            area7=Placeholder(name="area7"),
            area8=Placeholder(name="area8"),
        )
