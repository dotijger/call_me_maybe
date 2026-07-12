#!/usr/bin/env python3

import sys
import json
from src.error import ParsingError


def parse_flags() -> dict[str, str]:
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
    if check_flags(flags):
        return flags
    raise ParsingError("Flag definition is incorrect or incomplete.")


def check_flags(flags: dict[str, str]) -> bool:
    allowed = ["--functions_definition", "--input", "--output"]
    required = 0
    for name in flags.keys():
        if name not in allowed:
            return False
        if name == "--functions_definition":
            required = 1
    if not required:
        return False
    return True


def prompt_parsing(file: str) -> list[str]:
    prompts = []
    with open(file) as f:
        raw_prompts = json.load(f)
    for prompt in raw_prompts:
        for key, value in prompt.items():
            prompts.append(value)
    return prompts


def import_prompts() -> list[str]:
    try:
        flags = parse_flags()
    except ParsingError as e:
        print(e)
        sys.exit(1)
    input_file = flags.get("--input")
    if input_file:
        prompts = prompt_parsing(input_file)
    else:
        prompts = prompt_parsing("data/input/function_calling_tests.json")
    return prompts


def test() -> None:
    try:
        flags = parse_flags()
    except ParsingError as e:
        print(e)
        sys.exit(1)
    print(flags)


if __name__ == "__main__":
    import_prompts()
