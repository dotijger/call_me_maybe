import sys
import json
from src.error import ParsingError
from pydantic import BaseModel, model_validator
from typing import Self


class Parsing(BaseModel):
    input: str = "data/input/function_calling_tests.json"
    output: str = "data/output/function_calls.json"
    func_def: str = "data/input/functions_definition.json"

    @model_validator(mode="after")
    def load(self) -> Self:
        try:
            flags = self._parse_flags()
        except (ParsingError, FileNotFoundError) as e:
            print(
                f"Oops... There seems to be a\
 problem with your input files: {e}"
            )
            sys.exit(1)
        if flags.get("--input"):
            self.input = flags["--input"]
        if flags.get("--output"):
            self.output = flags["--output"]
        if flags.get("--functions_definition"):
            self.func_def = flags["--functions_definition"]
        return self

    def _parse_flags(self) -> dict[str, str]:
        flags = {}
        arguments = []
        if len(sys.argv) <= 2:
            raise ParsingError(
                "No functions definition specified, exiting program"
            )
        if len(sys.argv) >= 3:
            arguments = sys.argv.copy()
            while len(arguments) > 1:
                flags[arguments.pop(-1)] = arguments.pop(-1)
        try:
            self._check_flags(flags)
            return flags
        except ParsingError as e:
            raise ParsingError(f"{e}")

    @staticmethod
    def _check_flags(flags: dict[str, str]) -> None:
        allowed = ["--functions_definition", "--input", "--output"]
        required = 0
        for name in flags.keys():
            if name not in allowed:
                raise ParsingError(
                    f"{name} is not an allowed flag to run this program with."
                )
            if name == "--functions_definition":
                required = 1
        for flag, path in flags.items():
            if flag != "-output":
                try:
                    with open(path) as f:
                        _ = json.load(f)
                except json.JSONDecodeError as e:
                    raise ParsingError(
                        f"JSON given by {flag} cannot be read: {e}"
                    )
        if not required:
            raise ParsingError(
                "Required flag '--functions_definition' not defined"
            )


def input_parsing(file: str) -> list[str]:
    prompts = []
    with open(file) as f:
        raw_prompts = json.load(f)
    for prompt in raw_prompts:
        for key, value in prompt.items():
            prompts.append(value)
    return prompts
