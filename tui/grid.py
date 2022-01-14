from textual.views import GridView

from tui.widgets.params import Params
from tui.widgets.summary import Summary
from tui.widgets.selector import Selector
from tui.widgets.devices import DeviceStats
from tui.widgets.checkpoints import Checkpoints


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
            area1=DeviceStats(),
            area2=Params(),
            area3=Summary(),
            area4=Checkpoints("tb_logs"),
            area5=Selector("model", ["BERT", "RoBERTa"]),
            area6=Selector(
                "dataset",
                ["tweet_eval", "amazon_reviews", "another_dataset", "and_another"],
            ),
            area7=Selector("task", ["train", "test", "predict"]),
            area8=Selector("TBD", [""]),
        )
