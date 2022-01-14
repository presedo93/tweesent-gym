from textual.message import Message


class EnterPressed(Message, bubble=True):
    pass


class EnterCommand(Message, bubble=True):
    pass
