import logging
import sys
from pydantic import BaseModel, PrivateAttr
from typing import Any
from collections import deque


class HistoryHandler(logging.Handler):
    """A silent logging handler that keeps a full in-memory history.

    Every formatted record is appended to ``records`` instead of being
    printed or written anywhere, so the log history can be inspected or
    replayed later without producing terminal output.
    """

    def __init__(self) -> None:
        """Initializes the handler with an empty record history.

        Returns:
            None
        """
        super().__init__()
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Formats and stores a log record in the in-memory history.

        Args:
            record (logging.LogRecord): The log record to store.

        Returns:
            None
        """
        self.records.append(self.format(record))


class VisualHandler(logging.Handler):
    """A logging handler that redraws a scrolling terminal dashboard.

    Keeps only the most recent ``max_msg`` formatted records and
    repaints the terminal (header plus buffered messages) each time a
    new record is emitted, giving a live, ANSI-colored view of program
    activity.
    """

    HEADER: str = ""

    def __init__(self, max_msg: int = 7) -> None:
        """Initializes the handler and loads the dashboard header text.

        Args:
            max_msg (int): The maximum number of recent messages kept
                and displayed at once. Defaults to ``7``.

        Returns:
            None
        """
        super().__init__()
        self.max_msg = max_msg
        self.buffer: deque[str] = deque(maxlen=max_msg)
        with open("src/assets/header.txt", "r") as header:
            for line in header:
                self.HEADER += line

    def emit(self, record: logging.LogRecord) -> None:
        """Buffers a formatted record and redraws the dashboard.

        Args:
            record (logging.LogRecord): The log record to display.

        Returns:
            None
        """
        self.buffer.append(self.format(record))
        self._draw()

    def _draw(self) -> None:
        """Redraws the terminal with the header and buffered messages.

        Falls back to printing only the latest message when standard
        output is not attached to a terminal (e.g. when output is
        redirected to a file).

        Returns:
            None
        """
        if not sys.stdout.isatty():
            print(self.buffer[-1])
            return

        print("\033[2J\033[H", end="")
        print(self.HEADER)
        for line in self.buffer:
            print(line)
        for no_message in range(self.max_msg - len(self.buffer)):
            print()


class Logger(BaseModel):
    """A pydantic wrapper around Python's standard ``logging`` module.

    Always attaches a silent file handler, and optionally attaches a
    live-updating terminal dashboard handler when ``visual`` is
    enabled.

    Attributes:
        name (str): The name of the underlying ``logging.Logger``.
        level (int): The minimum logging level that will be handled.
        visual (bool): Whether to enable the live terminal dashboard.
        filepath (str): The path of the log file to write to.
    """

    name: str = "call_me_maybe"
    level: int = logging.INFO
    _logger: logging.Logger = PrivateAttr()
    visual: bool = False
    filepath: str = "log"

    def model_post_init(self, __context: Any) -> None:
        """Configures the underlying logger and attaches its handlers.

        Creates the standard-library logger, attaches a file handler
        (always), and attaches a ``VisualHandler`` (only when
        ``visual`` is ``True``).

        Args:
            __context (Any): The pydantic post-init context, unused.

        Returns:
            None
        """
        self._logger = logging.getLogger(self.name)
        self._logger.setLevel(self.level)
        format = logging.Formatter(fmt="%(message)s")
        if self.visual:
            visual_handler = VisualHandler()
            visual_handler.setLevel(self.level)
            visual_handler.setFormatter(format)
            self._logger.addHandler(visual_handler)
        file_handler = logging.FileHandler(self.filepath, mode="w")
        file_handler.setLevel(self.level)
        file_handler.setFormatter(format)
        self._logger.addHandler(file_handler)

    def log(self, level: int, message: str) -> None:
        """Logs a message at the given severity level.

        Args:
            level (int): The logging level (e.g. ``logging.INFO``).
            message (str): The message to log.

        Returns:
            None
        """
        self._logger.log(level, message)
