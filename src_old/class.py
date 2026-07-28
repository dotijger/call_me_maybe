from pydantic import BaseModel
from typing import Any


{"prompt": "", "name": "", "parameters": {"key": "value"}}


class JSONStateMachine(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, Any]

    def __init__(self):
        self.states = {
            "open": self.open_state,
            "key": self.key_state,
            "colon": self.colon_state,
            "value": self.value_state,
            "close": self.close_state,
            "comma": self.comma_state,
        }
