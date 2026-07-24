#!/usr/bin/env python3

from new.error import EncodeError, DecodeError
from new.parsing import Path
from new.function_parser import input_parsing
from pydantic import BaseModel, model_validator, ConfigDict
from new.classes import Trie, TrieNode, Vocab, JSONraw, OutputDict
from llm_sdk.llm_sdk import Small_LLM_Model
from typing import Any
from new.state import ParameterState, is_candidate_allowed
import json
import numpy as np
# decoder = ConstrainedDecoder(path=path,llm=Small_LLM_Model(),trie_vocab={})


class ConstrainedDecoder(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
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
    path: Path | None = None
    llm: Small_LLM_Model  # insert model name as first param, if blank default = qwen
    prompts: list[str] | None = None
    functions: list[JSONraw] | None = None
    function_names: list[str] | None = None
    trie_functions: Trie | None = None
    trie_vocab: Vocab | None = None
    llm_vocab: Vocab | None = None
    param_schema: dict[str, list[str]] | None = None
    output: list[str] | None = None
    nl_input_ids: list[int] | None = None

    @model_validator(mode="after")
    def setup(self) -> None:
        if self.path is None:
            self.path = Path()
        # json_vocab:
        self.trie_vocab = Vocab(vocab=self.vocab_json)
        # parsing the prompts
        self.prompts = input_parsing(self.path.input)

        # parsing the function definitions
        with open(self.path.func_def) as f:
            self.functions = json.load(f)
        self.function_names = []
        for function in self.functions:
            self.function_names.append(function.get("name"))

        # creating the trie structure for the possible function names
        self.trie_functions = Trie(vocab=self.trie_vocab, entries=self.function_names)

        # loading the llm vocab
        if self.llm_vocab is None:
            vocab_path = self.llm.get_path_to_vocab_file()
            with open(vocab_path) as v:
                vocab = json.load(v)
            # as the llm vocab is dict[str, int] and Vocab() accepts dict[int, str]
            self.llm_vocab = Vocab(
                vocab={k: v for v, k in vocab.items()}, inverted=vocab
            )

        # loading paramater schema lookup dictionary
        self.param_schema = self._loading_parameters()

        # setting up default natural language prompt (for fn_name)
        natural_language = " You are a function-calling assistant. \
        Given a set of allowed functions, answer the prompt by \
        responding with the function name. \
        Allowed functions and their descriptions: "
        for function in self.functions:
            natural_language += f"Name: '{function.get('name')}', description: {function.get('description')}."
        text = self._replace_space(natural_language)
        self.nl_input_ids = self._encode(text)

        return self

    def _loading_parameters(self) -> dict[str, dict[str, str]]:
        param_schema = {}
        for function in self.functions:
            parameters = []
            try:
                for pname, ptype in function.get("parameters").items():
                    parameters.append((pname, ptype["type"]))
            except KeyError:
                raise KeyError
            param_schema[function.get("name")] = parameters
        return param_schema

    def run(self) -> None:
        self.output = []
        for prompt in self.prompts:
            self.output.append(self._process_prompt(prompt))
        print(self.output)

    def _process_prompt(self, prompt: str) -> str:
        output = '{"prompt": "' + prompt + '", "name": "'
        name = self._generate_name(prompt)
        output = output + name + '", "parameters": {'
        parameters = self.param_schema[name]
        print(parameters)
        amount = len(parameters)
        i = 1
        info = (name, amount)
        for key, value in parameters:
            output += f'"{key}": '
            parameter = self._get_parameter(prompt, info, value, i)
            output += f"{parameter}"
            if i != amount:
                output += " "
            i += 1
        output += "}"
        return output

    def _generate_name(self, prompt: str) -> str:
        generated = ""
        generating = True
        input_ids = []
        not_allowed = []
        prompts = f"Answer this prompt: {prompt}"
        text = self._replace_space(prompts)
        input_ids += self.nl_input_ids + self._encode(text)
        print(repr(self._decode(input_ids)))
        print(type(input_ids))
        while generating is True:
            print(f"step, generated so far: {generated!r}", flush=True)
            logits = np.array(self.llm.get_logits_from_input_ids(input_ids))
            allowed = self._allowed(generated, "name", False, None)
            mask = self._get_mask(logits, allowed, not_allowed)
            masked = logits + mask
            next_id = np.argmax(masked)
            if self.trie_functions.search(generated):
                generating = False
            else:
                temp_gen = generated + self.llm_vocab.vocab.get(next_id)
                if self.trie_functions.is_prefix(temp_gen):
                    not_allowed = []
                    input_ids.append(next_id)
                    generated += self.llm_vocab.vocab.get(next_id)
                else:
                    not_allowed.append(next_id)
        return generated

    def _allowed(
        self, generated: str, kind: str, end: bool, prompt: str | None
    ) -> list[int]:
        allowed = []
        if kind == "name":
            for value in self.llm_vocab.vocab.values():
                if self.trie_functions.is_prefix(generated + value):
                    allowed.append(self.llm_vocab.inverted.get(value))
                elif self.trie_functions.search(generated + value):
                    allowed.append(self.llm_vocab.inverted.get(value))
                else:
                    continue
        elif kind == "string":
            if generated == "":
                state = ParameterState.STRING_START
                for value in self.llm_vocab.vocab.values():
                    if is_candidate_allowed(state, generated, value, end, prompt):
                        allowed.append(self.llm_vocab.inverted.get(value))
            else:
                state = ParameterState.STRING_MID
                for value in self.llm_vocab.vocab.values():
                    if is_candidate_allowed(state, generated, value, end, prompt):
                        allowed.append(self.llm_vocab.inverted.get(value))
        elif kind == "number":
            if generated == "":
                state = ParameterState.NUM_START
                for value in self.llm_vocab.vocab.values():
                    if is_candidate_allowed(state, generated, value, end, prompt):
                        allowed.append(self.llm_vocab.inverted.get(value))
            else:
                if self._check_dots(generated):
                    state = ParameterState.NUM_MID_DOT
                    for value in self.llm_vocab.vocab.values():
                        if is_candidate_allowed(state, generated, value, end, prompt):
                            allowed.append(self.llm_vocab.inverted.get(value))
                else:
                    state = ParameterState.NUM_MID_NO_DOT
                    for value in self.llm_vocab.vocab.values():
                        if is_candidate_allowed(state, generated, value, end, prompt):
                            allowed.append(self.llm_vocab.inverted.get(value))
        else:
            print("Undefined token request, not allowing any IDs")
        return allowed

    def _get_parameter(
        self, prompt: str, function: tuple[str, int], kind: str, count: int
    ) -> str:
        llm_prompt = self._generate_param_prompt(function, count, prompt, kind)
        text = self._replace_space(llm_prompt)
        print(text)
        input_ids = self._encode(text)
        parameter = ""
        fn_words = function[0].split("_")
        prompt_words = prompt.lower().split(" ")
        parameter_prompt = ""
        for word in prompt_words:
            if word not in fn_words:
                parameter_prompt = parameter_prompt + word + " "
        # regex = self._check_regex(function)
        if kind == "string":
            parameter = self._get_string(
                input_ids, (function[1] == count), parameter_prompt
            )
            return parameter
        elif kind == "number":
            parameter = self._get_number(
                input_ids, (function[1] == count), parameter_prompt
            )
        return parameter

    def _get_number(
        self, input_ids: list[int], is_last_parameter: bool, prompt: str
    ) -> str:
        generated = ""
        generating = True
        terminator = " " if is_last_parameter else ","
        while generating is True:
            print(f"step, generated so far: {generated!r}", flush=True)
            logits = np.array(self.llm.get_logits_from_input_ids(input_ids))
            allowed = self._allowed(generated, "number", is_last_parameter, prompt)
            if not allowed:
                generating = False
                break
            mask = self._get_mask(logits, allowed, None)
            masked = logits + mask
            next_id = np.argmax(masked)
            if generated.endswith(terminator) or len(generated) > 20:
                generating = False
            else:
                generated += self.llm_vocab.vocab.get(next_id)
                input_ids.append(next_id)
        if is_last_parameter:
            generated = generated.rstrip(" ") + "}"
        return generated

    def _get_string(
        self, input_ids: list[int], is_last_parameter: bool, prompt: str
    ) -> str:
        generated = '"'
        generating = True
        terminator = "}" if is_last_parameter else ","
        end = f'"{terminator}'
        input_ids += self._encode('"')
        while generating is True:
            print(f"step, generated so far: {generated!r}", flush=True)
            logits = np.array(self.llm.get_logits_from_input_ids(input_ids))
            allowed = self._allowed(generated, "string", is_last_parameter, prompt)
            if not allowed:
                generating = False
                break
            mask = self._get_mask(logits, allowed, None)
            masked = logits + mask
            next_id = np.argmax(masked)
            if generated.endswith(end):
                generating = False
            else:
                generated += self.llm_vocab.vocab.get(next_id)
                input_ids.append(next_id)
        generated = self._replace_g(generated)
        return generated

    # masking helpers

    def _get_mask(
        self, logits: list[float], ids: list[int], non: list[int] | None
    ) -> np.typing.ArrayLike:
        # an id is also its 'index' in the vocabulary / the key
        mask = np.full(len(logits), -np.inf)
        if non is None:
            for id in ids:
                mask[id] = 0
            return mask
        for id in ids:
            if id not in non:
                mask[id] = 0
        return mask

    # encoding + helpers
    def _encode(self, string: str) -> list[int]:
        ids = []
        possible_ids = []
        tokenized = ""
        i = 0
        sub = string[i]
        pop = 0
        while tokenized != string:
            tmp = self._find_match(sub)
            if len(possible_ids) > 0:
                if possible_ids[-1] == tmp:
                    pop = 1
            if tmp == -1 or pop:
                if self._is_prefix_string(sub, self.llm_vocab.vocab):
                    i += 1
                    if i < len(string):
                        sub += string[i]
                        continue
                if len(possible_ids) == 0:
                    print(tokenized, string)
                    raise EncodeError(
                        "String cannot be encoded, vocabulary insufficient"
                    )
                longest = self._find_longest_match(possible_ids)
                ids.append(longest)
                possible_ids = []
                tokenized += self.llm_vocab.vocab.get(longest)
                if tokenized == string:
                    return ids
                i = string.find(tokenized) + len(tokenized)
                sub = string[i]
                pop = 0
            else:
                possible_ids.append(tmp)
                if tokenized + sub == string:
                    longest = self._find_longest_match(possible_ids)
                    ids.append(longest)
                    return ids
                else:
                    i += 1
                    if i < len(string):
                        sub += string[i]
        longest = self._find_longest_match(possible_ids)
        ids.append(longest)
        return ids

    def _find_match(self, string: str) -> int:
        id = -1
        for i in range(self.llm_vocab.size):
            if self.llm_vocab.vocab.get(i) == string:
                id = i
        return id

    def _find_longest_match(self, ids: list[int]) -> int:
        max = 0
        longest = -1
        for id in ids:
            if len(self.llm_vocab.vocab.get(id)) > max:
                max = len(self.llm_vocab.vocab.get(id))
                longest = id
        return longest

    # decoder
    def _decode(self, ids: list[int]) -> str:
        decoded = ""
        for id in ids:
            next_str = self.llm_vocab.vocab.get(id)
            if next_str is None:
                raise DecodeError("Given ID is not in LLM vocab")
            decoded += next_str
        return decoded

    # parameter prompt generator
    def _generate_param_prompt(
        self, info: tuple[str, int], count: int, prompt: str, kind: str
    ) -> str:
        parameters = self.param_schema[info[0]]
        param_to_extract = parameters[count - 1][0]
        prompt = f"User prompt: {prompt} \
        Function being called: {info[0]} \
        Parameter to extract: {param_to_extract}. \
        Extract the value of {param_to_extract} from the user prompt. \
        Copy the {kind} parameter from the prompt: \
        {prompt}. {param_to_extract} = "
        return prompt

    @staticmethod
    def _is_prefix(small: str, big: str) -> bool:
        if len(small) > len(big) or len(small) == 0:
            return False
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

    @staticmethod
    def _allowed_tokens(text: str) -> list[str]:
        return text.split(" ")

    @staticmethod
    def _replace_space(text: str) -> str:
        return text.replace(" ", "Ġ")

    @staticmethod
    def _replace_g(text: str) -> str:
        if text[0] == "Ġ":
            return text[1:].replace("Ġ", " ")
        return text.replace("Ġ", " ")

    @staticmethod
    def _check_dots(text: str) -> bool:
        dots = 0
        for char in text:
            if char == ".":
                dots += 1
        return dots
