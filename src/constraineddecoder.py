#!/usr/bin/env python3

from src.parsing import Parsing, input_parsing
from pydantic import BaseModel, model_validator, ConfigDict, Field
from src.error import ParameterError, EncodeError
from src.classes import Vocab, OutputDict, Color
from llm_sdk.llm_sdk import Small_LLM_Model
from typing import Self, Any
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
    llm: Small_LLM_Model
    # insert model name as first param, if blank default = qwen
    prompts: list[str] = []
    functions: list[dict[str, Any]] = []
    function_names: list[str] = []
    llm_vocab: Vocab = Field(default_factory=lambda: Vocab(vocab={}))
    coder: Coder = Coder(llm_vocab=Vocab(vocab={}))
    param_schema: dict[str, list[tuple[str, str]]] = {}
    nl_input_ids: list[int] = []
    output_dict: OutputDict = {"prompt": "", "name": "", "parameters": {}}
    output_list: list[OutputDict] = []
    REGEX_EXAMPLES: str = "Examples: extract digits or numbers -> \\d+ ; \
extract vowels -> [aeiouAEIOU] ; extract whitespace -> \\s+ ; \
match the whole word 'cat' -> \\bcat\\b ; extract letters -> [a-zA-Z]+ ."
    REPLACEMENT_EXAMPLES: str = "Examples: replace with NUMBERS -> NUMBERS ; \
replace with asterisks -> * ; replace with dog -> dog ."

    @model_validator(mode="after")
    def setup(self) -> Self:
        # parsing the prompts
        self.prompts = input_parsing(self.path.input)
        # parsing the function definitions
        with open(self.path.func_def) as f:
            self.functions = json.load(f)
        self.function_names = []
        for function in self.functions:
            self.function_names.append(function["name"])
        # loading the llm vocab
        vocab_path = self.llm.get_path_to_vocab_file()
        with open(vocab_path) as v:
            vocab = json.load(v)
        # as the llm vocab is dict[str, int] and
        # Vocab() accepts dict[int, str]
        self.llm_vocab = Vocab(
            vocab={k: v for v, k in vocab.items()}, inverted=vocab
        )
        self.coder = Coder(llm_vocab=self.llm_vocab)
        # loading paramater schema lookup dictionary
        self.param_schema: dict[str, list[tuple[str, str]]] = (
            self._loading_parameters()
        )

        # setting up default natural language prompt (for fn_name)
        natural_language = " You are a function-calling assistant. \
        Given a set of allowed functions, answer the prompt by \
        responding with the function name. \
        Allowed functions and their descriptions: "
        for function in self.functions:
            natural_language += f"Name: '{function.get('name')}', \
                                description: {function.get('description')}."
            print(f"Function imported: {function.get('name')}")
        text = replace_space(natural_language)
        self.nl_input_ids = self.coder.encode(text)
        self._print_input()
        return self

    def _loading_parameters(self) -> dict[str, list[tuple[str, str]]]:
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
            self.output_list.append(self.output_dict)
        self._output_to_json()

    def _process_prompt(self, prompt: str) -> None:
        print(Color.DARK_GRAY.value + f"\nProcessing prompt: {prompt}...\n")
        namegen = NameGenerator(
            function_names=self.function_names,
            llm_vocab=self.llm_vocab,
            llm_prompt=list(self.nl_input_ids),
            coder=self.coder,
        )
        print("\nFunction selection...\n")
        function_prompt = (
            f"Which function should be used to answer the following: {prompt}"
        )
        function_name = namegen.generate(self.llm, function_prompt)
        print(Color.GREEN.value + f"\nFunction selected: {function_name}!")
        parameters = self.param_schema[function_name]
        amount = len(parameters)
        i = 1
        info = (function_name, amount)
        paramdict: dict[str, Any] = {}
        lexicon = self._prepare_lexicon(prompt, info[0])
        print(
            Color.BLUE.value
            + "\nExtracting parameter values...\n"
            + Color.DARK_GRAY.value
        )
        for key, value in parameters:
            llm_prompt = self._prompt(info, i, prompt, value, paramdict)
            text = replace_space(llm_prompt)
            try:
                input_ids = self.coder.encode(text)
            except EncodeError:
                input_ids = self.llm.encode(llm_prompt).tolist()
            used = self._remove_used_parameters(lexicon, paramdict)
            if len(used) < len(lexicon) and used != "":
                lexicon = used
            if value == "boolean":
                boolgen = self._get_bool_paramgen()
            else:
                try:
                    paramgen = self._get_parameter_generator(key, value)
                except ParameterError as e:
                    print(
                        Color.RED.value
                        + f"Unsupported parameter type {value} in \
                        {function_name}: {e}"
                        + Color.RESET.value
                    )
                    continue
            try:
                if value == "boolean":
                    encoded_prompt = self.coder.encode(lexicon)
                    parameter = boolgen.generate(
                        self.llm, encoded_prompt, prompt
                    )
                elif (
                    key == "regex" or key == "replacement"
                ) and value == "string":
                    parameter = paramgen.generate(
                        input_ids, (amount == i), prompt
                    )
                else:
                    parameter = paramgen.generate(
                        input_ids, (amount == i), lexicon
                    )
            except EncodeError as e:
                print(
                    Color.RED.value
                    + f"Parameter generation for {key}: {value} of \
                    {function_name} failed: {e}"
                    + Color.RESET.value
                )
            paramdict[key] = parameter
            i += 1
        for key, value in parameters:
            if value == "integer":
                if not is_number(paramdict[key]):
                    print(
                        Color.RED.value
                        + f"\nNo number value found for {key}: \
                        defaulting to null...\n"
                        + Color.RESET.value
                    )
                    paramdict[key] = None
                else:
                    tmp_int = int(paramdict[key])
                    paramdict[key] = tmp_int
            elif value == "number":
                if not is_number(paramdict[key]):
                    print(
                        Color.RED.value
                        + f"\nNo number value found for {key}: \
                        defaulting to null...\n"
                        + Color.RESET.value
                    )
                    paramdict[key] = None
                else:
                    tmp_flt = float(paramdict[key])
                    paramdict[key] = tmp_flt
            elif value == "boolean":
                if paramdict[key] == "True":
                    paramdict[key] = True
                else:
                    paramdict[key] = False
            else:
                continue
        print(
            Color.GREEN.value
            + f"\nParameters extracted: {paramdict}!\n"
            + Color.RESET.value
        )
        self.output_dict: OutputDict = {
            "prompt": prompt,
            "name": function_name,
            "parameters": paramdict,
        }

    def _prepare_lexicon(self, prompt: str, fn_name: str) -> str:
        terminator = (
            "'"
            if prompt.count("'") >= 2 and prompt.count("'") % 2 == 0
            else '"'
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

    def _get_parameter_generator(
        self, key: str, value: str
    ) -> BaseParameterGenerator:
        if key == "regex" or key == "replacement":
            return RegexParameterGenerator(
                llm=self.llm,
                llm_vocab=self.llm_vocab,
                coder=self.coder,
                is_pattern=(key == "regex"),
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
        elif value == "array":
            return BaseParameterGenerator(
                llm=self.llm, llm_vocab=self.llm_vocab, coder=self.coder
            )
        elif value == "object":
            return BaseParameterGenerator(
                llm=self.llm, llm_vocab=self.llm_vocab, coder=self.coder
            )
        else:
            return BaseParameterGenerator(
                llm=self.llm, llm_vocab=self.llm_vocab, coder=self.coder
            )

    def _get_bool_paramgen(self) -> BoolParameterGenerator:
        return BoolParameterGenerator(
            function_names=["True", "False"],
            llm_vocab=self.llm_vocab,
            coder=self.coder,
        )

    # parameter prompt generator
    def _prompt(
        self,
        info: tuple[str, int],
        count: int,
        prompt: str,
        kind: str,
        paramdict: dict[str, str],
    ) -> str:
        parameters = self.param_schema[info[0]]
        param_to_extract = parameters[count - 1][0]
        prompt = f"User prompt: {prompt} \
        Function being called: {info[0]}"
        if len(paramdict.values()) > 0:
            prompt += f"Do not extract these: {paramdict}"
        else:
            prompt += "No parameters extracted yet."
        prompt += f"Parameter to extract: {param_to_extract}. \
        Extract the value of {param_to_extract} from the user prompt."
        if param_to_extract == "regex":
            prompt += self.REGEX_EXAMPLES
        elif param_to_extract == "replacement":
            prompt += self.REPLACEMENT_EXAMPLES
        prompt += f"Copy the {kind} parameter from the prompt: \
        {prompt}. {param_to_extract} = "
        return prompt

    @staticmethod
    def _allowed_tokens(text: str) -> list[str]:
        return text.split(" ")

    def _remove_used_parameters(
        self, text: str, paramdict: dict[str, str]
    ) -> str:
        available = text.split(" ")
        if len(paramdict.items()) == 0:
            return ""
        used_nbr = []
        used_str = []
        for value in paramdict.values():
            if value is None:
                continue
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
        return dots != 0

    def _output_to_json(self) -> None:
        print(
            Color.BLUE.value
            + f"Outputting responses to JSON to {self.path.output}..."
        )
        output_path = Path(self.path.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path.output, "w") as jfile:
            json.dump(self.output_list, jfile, indent=2)
        print(
            Color.GREEN.value
            + "Output complete!\n"
            + Color.MAGENTA.value
            + "For more prompting, call me again, maybe?\n<End>\n"
        )

    def _print_input(self) -> None:
        print(
            Color.MAGENTA.value + "Initializing program..." + Color.RESET.value
        )
        print(
            Color.GREEN.value
            + f"\nFunction definitions extracted from: {self.path.func_def}"
        )
        print(
            f"\nPrompts imported from: {self.path.input}" + Color.RESET.value
        )
