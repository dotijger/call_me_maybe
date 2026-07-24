#!/usr/bin/env python3

import json
from typing import TypedDict
from pydantic import BaseModel
from new.parsing import Path


class JSONraw(TypedDict):
    name: str
    description: str
    parameters: dict[str, dict[str, str]]
    returns: dict[str, str]


class Function(BaseModel):
    raw: JSONraw


def input_parsing(file: str) -> list[str]:
    prompts = []
    with open(file) as f:
        raw_prompts = json.load(f)
    for prompt in raw_prompts:
        for key, value in prompt.items():
            prompts.append(value)
    return prompts


def main() -> None:
    path = Path()
    path.load()
    prompts = input_parsing(path.input)


if __name__ == "__main__":
    main()
# make a list of function names + description + return to give to llm
# make a convenient dictionary to extract the parameters quickly
