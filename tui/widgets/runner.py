from rich.console import RenderableType

from textual.widget import Widget
from textual.reactive import Reactive

from tools.utils import default_args
from train import train


class Runner(Widget):
    once: Reactive[bool] = Reactive(True)

    def watch_visible(self, visible: bool) -> None:
        if visible and self.once:
            self.set_timer(1.5, self.launch)
            self.once = False

        if not visible:
            self.once = True

    def launch(self) -> None:
        args = default_args()
        args.gpus = 1
        args.metrics = "acc"
        args.fast_dev_run = True
        train(args)

    def render(self) -> RenderableType:

        return ""
