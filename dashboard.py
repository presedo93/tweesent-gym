from textual.app import App
from textual.reactive import Reactive
from textual.widgets import Placeholder

from tui.widgets.header import Header
from tui.widgets.command import Command
from tui.grid import DashGrid


class Dashboard(App):
    show_bar = Reactive(False)

    async def on_load(self) -> None:
        """Bind keys here."""
        await self.bind("ctrl+b", "toggle_sidebar", "Toggle sidebar")
        await self.bind("ctrl+c", "quit", "Quit")

    def watch_show_bar(self, show_bar: bool) -> None:
        """Called when show_bar changes."""
        self.bar.animate("layout_offset_x", 0 if show_bar else -40)

    def action_toggle_sidebar(self) -> None:
        """Called when user hits 'b' key."""
        self.show_bar = not self.show_bar

    async def on_mount(self) -> None:
        """Build layout here."""
        # Main view
        await self.view.dock(Header(), edge="top", size=3)
        await self.view.dock(Command(), edge="bottom", size=3)
        await self.view.dock(DashGrid(), edge="top")

        # Side bar
        self.bar = Placeholder(name="left")
        await self.view.dock(self.bar, edge="left", size=40, z=1)
        self.bar.layout_offset_x = -40


Dashboard.run()
