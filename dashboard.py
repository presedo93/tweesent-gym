from textual.app import App
from textual.reactive import Reactive

from tui.grid import DashGrid
from tui.widgets.header import Header
from tui.widgets.command import Command
from tui.widgets.runner import Runner

from tui.messages import EnterCommand


class Dashboard(App):
    show_runner: Reactive[bool] = Reactive(False)

    async def on_load(self) -> None:
        """Bind keys here."""
        await self.bind("ctrl+b", "toggle_runner", "Toggle Runner")
        await self.bind("ctrl+c", "quit", "Quit")

    def watch_show_runner(self, show_runner: bool) -> None:
        """Called when show_runner changes."""
        self.runner.animate("visible", True if show_runner else False)

    def action_toggle_runner(self) -> None:
        """Called when user hits 'b' key."""
        self.show_runner = not self.show_runner

    async def handle_enter_pressed(self, message) -> None:
        print(message)
        await self.params.post_message(EnterCommand(self))

    async def on_mount(self) -> None:
        """Build layout here."""
        # Main view
        self.params = DashGrid()
        await self.view.dock(Header(), edge="top", size=3)
        await self.view.dock(Command(), edge="bottom", size=3)
        await self.view.dock(self.params, edge="top")

        # Runner
        self.runner = Runner()
        await self.view.dock(self.runner, edge="left", z=1)
        self.runner.visible = False


Dashboard.run()
