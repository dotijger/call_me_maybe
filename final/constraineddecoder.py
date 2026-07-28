#!/usr/bin/env python3

from final.parsing import Path, input_parsing
from pydantic import BaseModel, model_validator, ConfigDict
from final.classes import Vocab, JSONraw, OutputDict
from llm_sdk.llm_sdk import Small_LLM_Model
from typing import Any
from final.helpers import replace_space, replace_g, get_mask
from final.coder import Coder
from final.namegen import NameGenerator
from final.paramgen import (
    BaseParameterGenerator,
    StringParameterGenerator,
    RegexParameterGenerator,
    IntegerParameterGenerator,
    NumberParameterGenerator,
    BoolParameterGenerator,
)
import json
import numpy as np
# decoder = ConstrainedDecoder(path=path,llm=Small_LLM_Model())


class ConstrainedDecoder(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    path: Path = Path()
    llm: Small_LLM_Model  # insert model name as first param, if blank default = qwen
    prompts: list[str] | None = None
    functions: list[JSONraw] | None = None
    function_names: list[str] | None = None
    llm_vocab: Vocab | None = None
    param_schema: dict[str, list[str]] | None = None
    output: OutputDict | None = None
    output_list: list[OutputDict] | None = None
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
        for prompt in self.prompts:
            self._process_prompt(prompt)
            self.output_list.append(self.output)
        print(self.output_list)
        # self._output_to_json()

    def _process_prompt(self, prompt: str) -> None:
        self.output["prompt"] = prompt
        namegen = NameGenerator(
            self.function_names, self.llm_vocab, self.nl_input_ids, self.coder
        )
        self.output["name"] = namegen.generate(prompt)
        parameters = self.param_schema[self.output.get("name")]
        print(parameters)
        amount = len(parameters)
        i = 1
        info = (self.output.get("name"), amount)
        for key, value in parameters:
            paramdict = {}
            llm_prompt = self._prompt(info, i, prompt, value)
            text = replace_space(llm_prompt)
            input_ids = self.coder.encode(text)
            if key == "regex":
                paramgen = RegexParameterGenerator(self.llm, self.llm_vocab, self.coder)
            elif value == "string":
                paramgen = StringParameterGenerator(
                    self.llm, self.llm_vocab, self.coder
                )
            elif value == "number":
                paramgen = NumberParameterGenerator(
                    self.llm, self.llm_vocab, self.coder
                )
            elif value == "integer":
                paramgen = IntegerParameterGenerator(
                    self.llm, self.llm_vocab, self.coder
                )
            elif value == "" or value == "NULL" or value == "null" or value is None:
                paramgen = BaseParameterGenerator(self.llm, self.llm_vocab, self.coder)
            elif value == "boolean":
                paramgen = BoolParameterGenerator(
                    ["True", "False"], self.llm, self.llm_vocab, self.coder
                )
            elif value == "array":
                ...
            elif value == "object":
                ...
            parameter = paramgen.generate(input_ids, (amount == i), prompt)
            paramdict[key] = parameter
            i += 1
        self.output["parameters"] = paramdict

    def _get_parameter(
        self, prompt: str, function: tuple[str, int], kv: tuple[str, str], count: int
    ) -> str:
        """
        prompt = the function calling test prompt
        fucntion = tuple containing
                    str: the fn_name that has been given to answer this prompt,
                    int: the amount of parameters for this fn_name to fill out
        kv = tuple containing key (name of parameter) value (type of parameter)
        count = index of the parameter being asked to generate (if count == function[1], last parameter)
        """
        llm_prompt = self._prompt(function, count, prompt, kv[1])
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
    def _prompt(self, info: tuple[str, int], count: int, prompt: str, kind: str) -> str:
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
        return " ".join(available)

    def _output_to_json(self) -> None:
        with open("json_output.json", "w") as jfile:
            json.dump(self.output_list, jfile, indent=2)
