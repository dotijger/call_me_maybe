#!/usr/bin/env python3

import json
from typing import TypedDict


class Node(TypedDict):
    lit: str
    choices: list[str]
    number: None
    string: None


class JSONraw(TypedDict):
    name: str
    description: str
    parameters: dict[str, dict[str, str]]
    returns: dict[str, str]


class JSONproc(TypedDict):
    name: str
    parameters: dict[str, str]


class JSONSchema:
    def __init__(self, function: JSONraw) -> None:
        self._raw = function
        self._name = function["name"]
        self._processed = self._clean(self._raw)

    @property
    def name(self) -> str:
        return self._name

    @property
    def schema(self) -> JSONproc:
        return self._processed

    @property
    def info(self) -> JSONraw:
        return self._raw

    @staticmethod
    def _clean(raw: JSONraw) -> JSONproc:
        parameters: dict[str, str] = {}
        try:
            for pname, ptype in raw.get("parameters").items():
                parameters[pname] = ptype["type"]
        except KeyError:
            raise KeyError
        return {"name": raw.get("name"), "parameters": parameters}


class JSONLoader:
    def __init__(self) -> None:
        self._raw_functions: list[JSONraw]

    def import_json(self, json: list[JSONraw]) -> None:
        self._raw_functions = json

    def load_functions(self) -> list[JSONSchema]:
        """Takes raw functions list and creates individual
        JSON schemas for the LLM to use.
        """
        functions = []
        for function in self._raw_functions:
            functions.append(JSONSchema(function))
        return functions


def schema_import(file: str) -> list[JSONSchema]:
    loader = JSONLoader()
    with open(file) as f:
        loader.import_json(json.load(f))
    return loader.load_functions()


def schema_creation(functions: list[JSONSchema]) -> list[list[dict[str, str]]]:
    schemas = []
    for function in functions:
        schemas.append(tree_creation(function))
    return schemas


def tree_creation(function: JSONSchema) -> list[dict[str, str]]:
    j_string = json.dumps(function.schema)
    # at index 10 the function name begins with f
    print(j_string)
    tree = []
    tree.append({"lit": '"name": "'})
    pos = j_string.find('"', 10)
    tree.append({"choice": j_string[10:pos]})
    tree.append({"lit": '", "parameters": {'})
    pos += len(tree[2]["lit"])  # position of the " of the first param
    pos2 = j_string.find(":", pos) + 2  # the plus 1 is the space after
    tree.append({"lit": j_string[pos:pos2]})
    if j_string[pos2 + 1] == "n":
        tree.append({"digit": ""})
    if j_string[pos2 + 1] == "s":
        tree.append({"string": ""})
    if j_string.find(",", pos2) == -1:
        tree.append({"lit": "}}"})
        return tree
    pos = j_string.find('"', pos2 + 2) + 1
    pos3 = j_string.find(":", pos) + 2
    tree.append({"lit": j_string[pos:pos3]})
    if j_string[pos3 + 1] == "n":
        tree.append({"digit": ""})
    if j_string[pos3 + 1] == "s":
        tree.append({"string": ""})
    tree.append({"lit": "}"})
    return tree


if __name__ == "__main__":
    lists = schema_import("functions_definitions.json")
    for function in lists:
        print(tree_creation(function))


# parameters: {param1: {type: str}}
# paramter = dict[str, dict[str, str]]
# paramemter[str]
