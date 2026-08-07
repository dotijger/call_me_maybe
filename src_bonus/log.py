import logging
import sys
from pydantic import BaseModel, PrivateAttr
from typing import Any
from collections import deque


class HistoryHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(self.format(record))


class VisualHandler(logging.Handler):
    HEADER: str = ""

    def __init__(self, max_msg: int = 7) -> None:
        super().__init__()
        self.max_msg = max_msg
        self.buffer: deque[str] = deque(maxlen=max_msg)
        with open("src/assets/header.txt", "r") as header:
            for line in header:
                self.HEADER += line

    def emit(self, record: logging.LogRecord) -> None:
        self.buffer.append(self.format(record))
        self._draw()

    def _draw(self) -> None:
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
    name: str = "call_me_maybe"
    level: int = logging.INFO
    _logger: logging.Logger = PrivateAttr()
    visual: bool = False
    filepath: str = "log"

    def model_post_init(self, __context: Any) -> None:
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
        self._logger.log(level, message)
