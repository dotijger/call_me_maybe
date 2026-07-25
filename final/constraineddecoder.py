#!/usr/bin/env python3

from final.error import EncodeError, DecodeError
from final.parsing import Path, input_parsing
from pydantic import BaseModel, model_validator, ConfigDict
from final.classes import Trie, Vocab, JSONraw, OutputDict
from llm_sdk.llm_sdk import Small_LLM_Model
from typing import Any
from final.state import ParameterState, is_candidate_allowed
from final.helpers import is_prefix_string, replace_space, replace_g, get_mask
from final.coder import Coder
import json
import numpy as np
# decoder = ConstrainedDecoder(path=path,llm=Small_LLM_Model(),trie_vocab={})


class ConstrainedDecoder(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    path: Path()
    llm: Small_LLM_Model  # insert model name as first param, if blank default = qwen
    prompts: list[str] | None = None
    functions: list[JSONraw] | None = None
    function_names: list[str] | None = None
    llm_vocab: Vocab | None = None
    param_schema: dict[str, list[str]] | None = None
    output: str | None = None
    output_list: list[str] | None = None
    nl_input_ids: list[int] | None = None
    coder: Coder | None = None

    @model_validator(mode="after")
    def setup(self) -> None:
        # parsing the prompts
        self.prompts = input_parsing(self.path.input)
        # parsing the function definitions
        with open(self.path.func_def) as f:
            self.functions = json.load(f)
        self.function_names = []
        for function in self.functions:
            self.function_names.append(function.get("name"))
        # loading the llm vocab
        if self.llm_vocab is None:
            vocab_path = self.llm.get_path_to_vocab_file()
            with open(vocab_path) as v:
                vocab = json.load(v)
            # as the llm vocab is dict[str, int] and Vocab() accepts dict[int, str]
            self.llm_vocab = Vocab(
                vocab={k: v for v, k in vocab.items()}, inverted=vocab
            )
        self.coder = Coder(llm_vocab=self.llm_vocab)
        # loading paramater schema lookup dictionary
        self.param_schema = self._loading_parameters()

        # setting up default natural language prompt (for fn_name)
        natural_language = " You are a function-calling assistant. \
        Given a set of allowed functions, answer the prompt by \
        responding with the function name. \
        Allowed functions and their descriptions: "
        for function in self.functions:
            natural_language += f"Name: '{function.get('name')}', description: {function.get('description')}."
        text = replace_space(natural_language)
        self.nl_input_ids = self.coder.encode(text)

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
        self.output_list = []
        # print(f"{self.param_schema=}")
        # print(f"{self.functions=}")
        # print(f"{self.prompts=}")
        for prompt in self.prompts:
            self._process_prompt(prompt)
            self.output_list.append(self.output)
        print(self.output_list)
        # self._output_to_json()

    def _process_prompt(self, prompt: str) -> None:
        self.output = '{"prompt": "' + prompt + '", "name": "'
        name = self._generate_name(prompt)
        self.output = self.output + name + '", "parameters": {'
        parameters = self.param_schema[name]
        print(parameters)
        amount = len(parameters)
        i = 1
        info = (name, amount)
        for key, value in parameters:
            self.output += f'"{key}": '
            parameter = self._get_parameter(prompt, info, (key, value), i)
            self.output += f"{parameter}"
            if i != amount:
                self.output += " "
            i += 1
        self.output += "}"

    def _generate_name(self, prompt: str) -> str:
        generated = ""
        generating = True
        input_ids = []
        not_allowed = []
        prompts = f"Answer this prompt: {prompt}"
        text = self._replace_space(prompts)
        input_ids += self.nl_input_ids + self.coder.encode(text)
        print(repr(self.coder.decode(input_ids)))
        print(type(input_ids))
        while generating is True:
            print(f"step, generated so far: {generated!r}", flush=True)
            logits = np.array(self.llm.get_logits_from_input_ids(input_ids))
            allowed = self._allowed(generated, "name", False, None)
            mask = get_mask(logits, allowed, not_allowed)
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
        self, prompt: str, function: tuple[str, int], kv: tuple[str, str], count: int
    ) -> str:
        llm_prompt = self._generate_param_prompt(function, count, prompt, kv[1])
        text = replace_space(llm_prompt)
        print(text)
        input_ids = self.coder.encode(text)
        parameter = ""
        fn_words = function[0].split("_")
        prompt_words = prompt.split(" ")
        parameter_prompt = ""
        for word in prompt_words:
            if word.lower() not in fn_words:
                parameter_prompt = parameter_prompt + word + " "
        # substrings = self._get_substring(prompt)
        # if len(substrings) == 0:
        # substrings = [w for w in parameter_prompt.split(" ") if w]
        # substrings = "\x00".join(substrings)
        # regex = self._check_regex(function)
        if kv[0] == "regex":
            parameter = self._get_string(input_ids, (function[1] == count), "regex")
            return parameter
        if kv[1] == "string":
            parameter = self._get_string(
                input_ids, (function[1] == count), parameter_prompt
            )
            return parameter
        elif kv[1] == "number":
            used = self._remove_used(parameter_prompt)
            if used != "":
                parameter_prompt = used
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
            mask = get_mask(logits, allowed, None)
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
        self,
        input_ids: list[int],
        is_last_parameter: bool,
        prompt: str,
    ) -> str:
        generated = '"'
        generating = True
        terminator = "}" if is_last_parameter else ","
        end = f'"{terminator}'
        input_ids += self.coder.encode('"')
        while generating is True:
            print(f"step, generated so far: {generated!r}", flush=True)
            logits = np.array(self.llm.get_logits_from_input_ids(input_ids))
            allowed = self._allowed(generated, "string", is_last_parameter, "regex")
            if not allowed:
                generating = False
                break
            mask = get_mask(logits, allowed, None)
            masked = logits + mask
            next_id = np.argmax(masked)
            if generated.endswith(end):
                generating = False
            else:
                generated += self.llm_vocab.vocab.get(next_id)
                input_ids.append(next_id)
        generated = replace_g(generated)
        return generated

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
    def _allowed_tokens(text: str) -> list[str]:
        return text.split(" ")

    @staticmethod
    def _check_dots(text: str) -> bool:
        dots = 0
        for char in text:
            if char == ".":
                dots += 1
        return dots

    def _remove_used(self, text: str) -> str:
        available = text.split(" ")
        words = self.output.split(" ")
        print(words)
        for i in range(len(words)):
            if words[i] == '"parameters":':
                cut = words[i:]
        if len(cut) == 0:
            return ""
        used = []
        for word in cut:
            if len(word) == 0:
                continue
            if word[-1] == ",":
                word = word[:-1]
                if word.isdigit():
                    used.append(word)
        if len(used) == 0:
            return ""
        for word in used:
            if word in available:
                available.remove(word)
        print(available)
        return " ".join(available)

    def _output_to_json(self) -> None:
        list_of_json = []
        for prompt in self.output_list:
            list_of_json.append(json.loads(prompt))
        with open("json_output.json", "w") as jfile:
            json.dump(list_of_json, jfile, indent=2)
