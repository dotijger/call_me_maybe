#!/usr/bin/env python3

from .parsing import Path, input_parsing
from pydantic import BaseModel, model_validator
from .classes import Trie, TrieNode, Vocab, JSONraw, OutputDict
from llm_sdk.llm_sdk import Small_LLM_Model
from typing import Any
import json

# decoder = ConstrainedDecoder(path=path,llm=Small_LLM_Model(),trie_vocab={})


class ConstrainedDecoder(BaseModel):
    vocab_json: dict[int, str] = {
        0: "_",
        1: "{",
        2: "}",
        3: ",",
        4: ":",
        5: "a",
        6: "b",
        7: "c",
        8: "d",
        9: "e",
        10: "f",
        11: "g",
        12: "h",
        13: "i",
        14: "j",
        15: "k",
        16: "l",
        17: "m",
        18: "n",
        19: "o",
        20: "p",
        21: "q",
        22: "r",
        23: "s",
        24: "t",
        25: "u",
        26: "v",
        27: "w",
        28: "x",
        29: "y",
        30: "z",
    }
    path: Path()
    llm: Small_LLM_Model()  # insert model name as first param, if blank default = qwen
    prompts: list[str] | None = None
    functions: list[JSONraw] | None = None
    function_names: list[str] | None = None
    trie_functions: Trie() | None = None
    trie_vocab: Vocab(vocab=vocab_json)
    llm_vocab: Vocab() | None = None
    param_schema: dict[str, list[str]] | None = None
    output: list[str] | None = None

    @model_validator(mode="false")
    def setup(self) -> None:
        # parsing the prompts
        self.prompts = input_parsing(self.path.input)

        # parsing the function definitions
        self.functions = json.load(self.path.functions_definition)
        for function in self.functions:
            self.function_names.append(function.get("name"))

        # creating the trie structure for the possible function names
        self.trie_functions = Trie(vocab=self.trie_vocab, entries=self.function_names)

        # loading the llm vocab
        vocab_path = self.llm.get_path_to_vocab_file()
        with open(vocab_path) as v:
            vocab = json.load(v)
        # as the llm vocab is dict[str, int] and Vocab() accepts dict[int, str]
        self.llm_vocab = Vocab(vocab={k: v for v, k in vocab.items()}, inverted=vocab)

        # loading paramater schema lookup dictionary
        self.param_schema = loading_parameters()

    def _loading_parameters(self) -> dict[str, dict[str, str]]:
        param_schema = {}
        for function in self.functions:
            parameters = []
            try:
                for pname, ptype in function.get("parameters").items():
                    parameters.append((pname, ptype["type"]))
            except KeyError:
                raise KeyError
            param_schema[function] = parameters
        return param_schema

    def run(self) -> None:
        for prompt in self.prompts:
            self.output.append(self._process_prompt(prompt))

    def _process_prompt(self, prompt: str) -> str:
        output = '{"prompt": "' + prompt + '", "name": "'
        name = self._get_name(prompt)
        output = output + name + '", "parameters": '
        parameters = self.param_schema[name]
        for i in range(parameters):
            output = output + '{"' + parameters[i][0] + '": '
            output += self._get_parameter(prompt, name, i)
            if i - 1 != len(parameters):
                output += ", "
            else:
                output += " }"

        # hardcode first bit
        # get function name (llm)
        # fill out function name + parameters
        # get # of parameters from paramschema (for loop over items)
        # get parameters from llm one by one (llm)

    def _generate_token(self, prompt: str) -> str:
        generated = ""

        return generated

    def _get_parameter(self, prompt: str, name: str, index: int) -> str:
        # need to add the terminator symbols to the generation
        # because otherwise it is very vague on whether or not it is done generating the param
        # so for strings the end is just the '"' symbol,
        # for numbers/others it is the ',' or '}'
        parameter = ""
        prototype = self.param_schema[name][index]
        if prototype[1] == "string":
            if len(parameter) == 0:
                # token has to start with '"' 
            else:
                # token can have only alphabetic char, and once " is read no more tokens are added
                allowed = '
            return parameter
        elif prototype[1] == "number":
            # token can start with '-' and have one '.', once both of those are read more than once not allowed
            # token ends with ',' or '}' depending on the index -> if last parameter, allowed = '}', otherwise ','
        elif prototype[1] == "":
            ...
        # output will let the llm know which parameter it is calculating
        return parameter

    @staticmethod
    def _is_prefix(small: str, big: str) -> bool:
        for i in range(len(small)):
            if small[i] == big[i]:
                continue
            else:
                return False
        return True

    def _is_prefix_string(self, s: str, valid: dict) -> bool:
        prefix = 0
        for value in valid.values():
            if self._is_prefix(s, value):
                prefix = 1
        return prefix == 1
