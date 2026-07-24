#!/usr/bin/env python3

import sys
import json
from src.error import ParsingError
from pydantic import BaseModel


class Path(BaseModel):
    input: str = "data/input/function_calling_tests.json"
    output: str = "data/output/function_calls.json"
    func_def: str = "data/input/functions_definition.json"

    def load(self) -> None:
        try:
            flags = parse_flags()
        except ParsingError as e:
            print(e)
            sys.exit(1)
        if flags.get("--input"):
            self.input = flags["--input"]
        if flags.get("--output"):
            self.output = flags["--output"]
        if flags.get("--functions_definition"):
            self.func_def = flags["--functions_definition"]


def parse_flags() -> dict[str, str]:
    flags = {}
    arguments = []
    if len(sys.argv) <= 2:
        raise ParsingError("No functions definition specified, exiting program")
    if len(sys.argv) >= 3:
        arguments = sys.argv.copy()
        while len(arguments) > 1:
            flags[arguments.pop(-1)] = arguments.pop(-1)
    try:
        if check_flags(flags):
            return flags
    except ParsingError as e:
        raise ParsingError(e)


def check_flags(flags: dict[str, str]) -> bool:
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


if __name__ == "__main__":
    import_prompts()
