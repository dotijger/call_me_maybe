import sys
import json
from final.error import ParsingError
from pydantic import BaseModel, model_validator


class Path(BaseModel):
    input: str = "data/input/function_calling_tests.json"
    output: str = "data/output/function_calls.json"
    func_def: str = "data/input/functions_definition.json"

    @model_validator(mode="after")
    def load(self) -> None:
        try:
            flags = self._parse_flags()
        except ParsingError as e:
            print(e)
            sys.exit(1)
        if flags.get("--input"):
            self.input = flags["--input"]
        if flags.get("--output"):
            self.output = flags["--output"]
        if flags.get("--functions_definition"):
            self.func_def = flags["--functions_definition"]

    def _parse_flags(self) -> dict[str, str]:
        flags = {}
        arguments = []
        if len(sys.argv) <= 2:
            raise ParsingError("No functions definition specified, exiting program")
        if len(sys.argv) >= 3:
            arguments = sys.argv.copy()
            while len(arguments) > 1:
                flags[arguments.pop(-1)] = arguments.pop(-1)
        try:
            if self._check_flags(flags):
                return flags
        except ParsingError as e:
            raise ParsingError(e)

    @staticmethod
    def _check_flags(flags: dict[str, str]) -> bool:
        allowed = ["--functions_definition", "--input", "--output"]
        required = 0
        for name in flags.keys():
            if name not in allowed:
                return False
            if name == "--functions_definition":
                required = 1
        for flag, path in flags.values():
            if flag != "--output":
                try:
                    with open(path) as f:
                        _ = json.load(f)
                except json.JSONDecodeError as e:
                    raise ParsingError(f"JSON given by {flag} cannot be read: {e}")
        if not required:
            return False
        return True


def input_parsing(file: str) -> list[str]:
    prompts = []
    with open(file) as f:
        raw_prompts = json.load(f)
    for prompt in raw_prompts:
        for key, value in prompt.items():
            prompts.append(value)
    return prompts
