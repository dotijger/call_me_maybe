#!/usr/bin/env python3

from functions_parser import schema_import, JSONSchema
import json


def build_prompt(prompt: str, functions: list[JSONSchema]) -> str:
    string_functions = ""
    for function in functions:
        string_functions += str(function.info)
    return (
        "allowed functions to use: "
        + string_functions
        + "\nuser request: "
        + prompt
    )


def build_schema(functions: list[JSONSchema]) -> list[str]:
    raw_schemas = []
    for function in functions:
        raw_schemas.append(json.dumps(function.schema))
    print(raw_schemas)
    print(type(raw_schemas))
    return raw_schemas


if __name__ == "__main__":
    lists = schema_import("functions_definitions.json")
    prompt = "what is 2 + 2"
    print(build_prompt(prompt, lists))
    build_schema(lists)
