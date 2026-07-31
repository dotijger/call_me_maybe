#!/usr/bin/env python3

from src.parsing import Parsing, input_parsing
from pydantic import BaseModel, model_validator, ConfigDict
from src.error import ParameterError, EncodeError
from src.classes import Vocab, JSONraw, OutputDict
from llm_sdk.llm_sdk import Small_LLM_Model
from typing import Self
from src.helpers import replace_space, is_number
from src.coder import Coder
from src.namegen import NameGenerator
from src.paramgen import (
    BaseParameterGenerator,
    StringParameterGenerator,
    RegexParameterGenerator,
    IntegerParameterGenerator,
    NumberParameterGenerator,
    BoolParameterGenerator,
)
import json
from pathlib import Path
# decoder = ConstrainedDecoder(path=path,llm=Small_LLM_Model())


class ConstrainedDecoder(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    path: Parsing = Parsing()
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
    def setup(self) -> Self:
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
        self._output_to_json()

    def _process_prompt(self, prompt: str) -> None:
        print(f"\nProcessing prompt: {prompt}...\n")
        self.output = {}
        self.output["prompt"] = prompt
        namegen = NameGenerator(
            function_names=self.function_names,
            llm_vocab=self.llm_vocab,
            llm_prompt=self.nl_input_ids,
            coder=self.coder,
        )
        print("\nFunction selection...\n")
        self.output["name"] = namegen.generate(self.llm, prompt)
        print(f"\nFunction selected: {self.output.get('name')}!")
        parameters = self.param_schema[self.output.get("name")]
        amount = len(parameters)
        i = 1
        info = (self.output.get("name"), amount)
        paramdict = {}
        lexicon = self._prepare_lexicon(prompt, info[0])
        print("\nExtracting parameter values...\n")
        for key, value in parameters:
            llm_prompt = self._prompt(info, i, prompt, value)
            text = replace_space(llm_prompt)
            try:
                input_ids = self.coder.encode(text)
            except EncodeError:
                input_ids = self.llm.encode(text)
            used = self._remove_used_parameters(lexicon, paramdict)
            if len(used) < len(lexicon) and used != "":
                lexicon = used
            try:
                paramgen = self._get_parameter_generator(key, value)
            except ParameterError as e:
                print(
                    f"Unsupported parameter type {value} in {self.output.get('name')}: {e}"
                )
                continue
            try:
                if value == "boolean":
                    parameter = paramgen.generate(self.llm, prompt)
                elif (key == "regex" or key == "replacement") and value == "string":
                    parameter = paramgen.generate(input_ids, (amount == i), prompt)
                else:
                    parameter = paramgen.generate(input_ids, (amount == i), lexicon)
            except EncodeError as e:
                print(
                    f"Parameter generation for {key}: {value} of {self.output.get('name')} failed: {e}"
                )
            paramdict[key] = parameter
            i += 1
        for key, value in parameters:
            if value == "integer":
                if not is_number(paramdict[key]):
                    print(
                        f"\nNo number value found for {key}: {paramdict[key]}: defaulting to 0...\n"
                    )
                    tmp = int(0)
                else:
                    tmp = int(paramdict[key])
                paramdict[key] = tmp
            elif value == "number":
                if not is_number(paramdict[key]):
                    print(
                        f"\nNo number value found for {key}: {paramdict[key]}: defaulting to 0...\n"
                    )
                    tmp = float(0)
                else:
                    tmp = float(paramdict[key])
                paramdict[key] = tmp
            elif value == "boolean":
                if paramdict[key] == "True":
                    paramdict[key] = True
                else:
                    paramdict[key] = False
            else:
                continue
        print(f"\nParameters extracted: {paramdict}!\n")
        self.output["parameters"] = paramdict

    def _prepare_lexicon(self, prompt: str, fn_name: str) -> str:
        terminator = (
            "'" if prompt.count("'") >= 2 and prompt.count("'") % 2 == 0 else '"'
        )
        fn_words = fn_name.split("_")
        prompt = prompt + " "
        result = []
        inside = False
        current_word = ""
        for char in prompt:
            if char == terminator:
                inside = not inside
                current_word += char
            elif char == " " and not inside:
                if current_word and current_word.lower() not in fn_words:
                    result.append(current_word)
                current_word = ""
            else:
                current_word += char
        return " ".join(result)

    def _get_parameter_generator(self, key: str, value: str) -> BaseParameterGenerator:
        if key == "regex" or key == "replacement":
            return RegexParameterGenerator(
                llm=self.llm, llm_vocab=self.llm_vocab, coder=self.coder
            )
        elif value == "string":
            return StringParameterGenerator(
                llm=self.llm, llm_vocab=self.llm_vocab, coder=self.coder
            )
        elif value == "number":
            return NumberParameterGenerator(
                llm=self.llm, llm_vocab=self.llm_vocab, coder=self.coder
            )
        elif value == "integer":
            return IntegerParameterGenerator(
                llm=self.llm, llm_vocab=self.llm_vocab, coder=self.coder
            )
        elif value == "boolean":
            return BoolParameterGenerator(
                function_names=["True", "False"],
                llm=self.llm,
                llm_vocab=self.llm_vocab,
                coder=self.coder,
            )
        elif value == "array":
            ...
        elif value == "object":
            ...
        else:
            return BaseParameterGenerator(
                llm=self.llm, llm_vocab=self.llm_vocab, coder=self.coder
            )

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

    def _remove_used_parameters(self, text: str, paramdict: dict[str, str]) -> str:
        available = text.split(" ")
        if len(paramdict.items()) == 0:
            return ""
        used_nbr = []
        used_str = []
        for value in paramdict.values():
            if is_number(value):
                used_nbr.append(value)
            else:
                used_str.append(value)
        for word in used_nbr:
            if word in available:
                available.remove(word)
        return " ".join(available)

    @staticmethod
    def _check_dots(text: str) -> bool:
        dots = 0
        for char in text:
            if char == ".":
                dots += 1
        return dots

    def _output_to_json(self) -> None:
        print(f"Outputting responses to JSON to {self.path.output}...")
        output_path = Path(self.path.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path.output, "w") as jfile:
            json.dump(self.output_list, jfile, indent=2)
        print("Output complete! For more prompting, call me again, maybe?\n<End>\n")
