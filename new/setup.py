#!/usr/bin/env python3

from .parsing import Path, input_parsing
from pydantic import BaseModel, model_validator
from .classes import Trie, TrieNode, Vocab, JSONraw, OutputDict
from llm_sdk.llm_sdk import Small_LLM_Model
from typing import Any
import json
import numpy as np
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

    def _generate_string(self, prompt: str) -> str:
        generated = ''
        generating = True
        input_ids = self._encode(prompt)
        while generating is True:
            logits = (self.llm.get_logits_from_input_ids(input_ids)).numpy()
            if len(generated) == 0:
                allowed = self._allowed_ids(0, 0)
            else:
                allowed = self._allowed_ids(1, 0)
            mask = self._get_mask(logits, allowed)
            masked = logits + mask
            next_id = argmax(masked)
            if self.llm_vocab.inverted.get(next_id)[-1] == '"':
                generating = False
            else:
                input_ids.append(next_id)
                generated += self.llm_vocab.inverted.get(next_id)
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

    # masking helpers

    def _allowed_ids(self, state: int, type: int) -> list[int]:
        # type = kind of token with our json output, 0 = string, 1 = number
        # state = where -> 0 = start, 1 = middle / end (so looking for end char as well, to see if last token)
        allowed = []
        if type == 0:
            if state == 0:
            # append all logits (translated) starting with '"' and only have alphabetic char after
            else:
            # append all logits that are only '"' (end char), or start with alphabetic and end with '"', or are just alphabetic
        elif type == 1:
            if state == 0:
            # append all logits starting with MAX 1 '-' and digit after, or digit, or one . in there
            elif state == 1:
            # state == 1 is not the last parameter
            # end char is ','
            else:
            # state ==2 is last parameter, so only end char is '}'
            # append all logits only digits, or ending with either ',' or '}'
        return allowed

    def _get_mask(self, logits: list[float], ids: list[int]) -> np.typing.ArrayLike:
        # an id is also its 'index' in the vocabulary / the key
        mask = np.full(self.llm_vocab.size, -np.inf)
        for id in ids:
            mask[id] = 0
        return mask

    # encoding + helpers
    def _encode(self, string: str) -> list[int]:
        ids = []
        possible_ids = []
        tokenized = ""
        i = 0
        sub = string[i]
        while tokenized != string:
            tmp = self._find_match(sub)
            if tmp == -1:
                if self._is_prefix_string(sub, self.llm_vocab.inverted):
                    i += 1
                    if i < len(string):
                        sub += string[i]
                    continue
                if len(possible_ids) == 0:
                    raise EncodeError("String cannot be encoded, vocabulary insufficient")
                longest = self._find_longest_match(possible_ids)
                ids.append(longest)
                possible_ids = []
                tokenized += self.llm_vocab.inverted.get(longest)
                i = string.find(tokenized) + len(tokenized)
                sub = ""
            else:
                if tokenized + sub == string:
                    possible_ids.append(tmp)
                    break
                possible_ids.append(tmp)
                i += 1
                sub += string[i]
        longest = self._find_longest_match(possible_ids)
        ids.append(longest)
        return ids

    def _find_match(self, string: str) -> int:
        id = -1
        for i in range(self.llm_vocab.size):
            if llm_vocab.inverted.get(i) == string:
                id = i
        return id

    def _find_longest_match(self, ids: list[int]) -> int:
        max = 0
        longest = -1
        for id in ids:
            if len(self.llm_vocab.inverted.get(id)) > max:
                max = len(self.llm_vocab.inverted.get(id))
                longest = id
        return longest

    # decoder
    def _decode(self, ids: list[int]) -> str:
        decoded = ""
        for id in ids:
            next_str = self.llm_vocab.inverted.get(id)
            if next_str is None:
                raise DecodeError("Given ID is not in LLM vocab")
            decoded += next_str
        return decoded

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
