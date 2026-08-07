#!/usr/bin/env python3

from src_bonus.parser import Parser
from src_bonus.log import Logger
from pydantic import BaseModel, model_validator, ConfigDict, Field
from src_bonus.error import ParameterError, EncodeError, LogError
from src_bonus.classes import Vocab, OutputDict, Color
from src_bonus.coder import Coder
from llm_sdk.llm_sdk import Small_LLM_Model
from typing import Self, Any
from src_bonus.helpers import is_number, replace_space
from src_bonus.namegen import NameGenerator
from src_bonus.paramgen import (
    BaseParameterGenerator,
    StringParameterGenerator,
    RegexParameterGenerator,
    IntegerParameterGenerator,
    NumberParameterGenerator,
    BoolParameterGenerator,
)
import json
import sys
import logging
# decoder = ConstrainedDecoder(path=path,llm=Small_LLM_Model())


class ConstrainedDecoder(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    path: Parser = Parser()
    llm: Small_LLM_Model
    log: Logger | None = None
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
substitute the word 'cat' -> 'cat' ; extract vowels -> [aeiouAEIOU] "
    REPLACEMENT_EXAMPLES: str = "Examples: replace with NUMBERS -> NUMBERS ; \
replace with asterisks -> * ; replace with dog -> dog ."

    @model_validator(mode="after")
    def setup(self) -> Self:
        # setup of the log
        if self.log is None:
            self.log = Logger(
                visual=self.path.args.visual
            )  # parsing the prompts
        try:
            self.prompts = self.path.parse_prompts()
        except ValueError as e:
            self.log.log(logging.CRITICAL, f"{e}")
            sys.exit(1)
        # parsing the function definitions
        with open(self.path.args.functions_definition) as f:
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
            self.log.log(
                logging.INFO, f"Function imported: {function.get('name')}"
            )
        natural_language += "For example: 'Greet john' -> fn_greet."
        text = replace_space(natural_language)
        self.nl_input_ids = self.coder.encode(text)
        self._print_input()
        return self

    def _loading_parameters(self) -> dict[str, list[tuple[Any, Any]]]:
        param_schema = {}
        for function in self.functions:
            parameters = []
            try:
                for pname, ptype in function["parameters"].items():
                    parameters.append((pname, ptype["type"]))
            except KeyError:
                raise KeyError
            name = function.get("name")
            if isinstance(name, str):
                param_schema[name] = parameters
        return param_schema

    def run(self) -> None:
        self.output_list = []
        for prompt in self.prompts:
            self._process_prompt(prompt)
            self.output_list.append(self.output_dict)
        self._output_to_json()

    def _process_prompt(self, prompt: str) -> None:
        if self.log is None:
            raise LogError(
                f"No logger found, cannot process prompt: {prompt}."
            )
        self.log.log(logging.INFO, f"\nProcessing prompt: {prompt}...\n")
        namegen = NameGenerator(
            function_names=self.function_names,
            llm_vocab=self.llm_vocab,
            llm_prompt=self.nl_input_ids,
            coder=self.coder,
            log=self.log,
        )
        self.log.log(logging.INFO, "\nFunction selection...\n")
        function_prompt = (
            f"Which function should be used to answer the following: {prompt}"
        )
        function_name = namegen.generate(self.llm, function_prompt)
        self.log.log(logging.INFO, f"\nFunction selected: {function_name}!")
        parameters = self.param_schema[function_name]
        amount = len(parameters)
        i = 1
        info = (function_name, amount)
        paramdict: dict[str, Any] = {}
        lexicon = self._prepare_lexicon(prompt, info[0])
        self.log.log(logging.INFO, "\nExtracting parameter values...\n")
        for key, value in parameters:
            llm_prompt = self._prompt(info, i, prompt, value, paramdict)
            text = replace_space(llm_prompt)
            input_ids = self.coder.encode(text)
            used = self._remove_used_parameters(lexicon, paramdict)
            if len(used) < len(lexicon) and used != "":
                lexicon = used
            if value == "boolean":
                boolgen = self._get_bool_paramgen()
            else:
                try:
                    paramgen = self._get_parameter_generator(key, value)
                except ParameterError as e:
                    self.log.log(
                        logging.WARNING,
                        f"Unsupported parameter type {value} in \
                        {function_name}: {e}",
                    )
                    continue
            try:
                if value == "boolean":
                    lexicon_g = replace_space(lexicon)
                    encoded_prompt = self.coder.encode(lexicon_g)
                    parameter: str | None = boolgen.generate(
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
                self.log.log(
                    logging.ERROR,
                    f"Parameter generation for {key}: {value} of \
                    {function_name} failed: {e}",
                )
            paramdict[key] = parameter
            i += 1
        for key, value in parameters:
            if value == "integer":
                if not is_number(paramdict[key]):
                    self.log.log(
                        logging.WARNING,
                        f"\nNo number value found for {key}: \
                        defaulting to null...\n",
                    )
                    paramdict[key] = None
                else:
                    tmp_int = int(paramdict[key])
                    paramdict[key] = tmp_int
            elif value == "number":
                if not is_number(paramdict[key]):
                    self.log.log(
                        logging.WARNING,
                        f"\nNo number value found for {key}: \
                        defaulting to null...\n",
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
        self.log.log(logging.INFO, f"\nParameters extracted: {paramdict}!\n")
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
        if self.log is None:
            raise LogError(
                "Logger not defined, unable to pass logger to parameter\
generator."
            )
        if key == "regex" or key == "replacement":
            return RegexParameterGenerator(
                llm=self.llm,
                llm_vocab=self.llm_vocab,
                coder=self.coder,
                log=self.log,
                is_pattern=(key == "regex"),
            )
        elif value == "string":
            return StringParameterGenerator(
                llm=self.llm,
                llm_vocab=self.llm_vocab,
                coder=self.coder,
                log=self.log,
            )
        elif value == "number":
            return NumberParameterGenerator(
                llm=self.llm,
                llm_vocab=self.llm_vocab,
                coder=self.coder,
                log=self.log,
            )
        elif value == "integer":
            return IntegerParameterGenerator(
                llm=self.llm,
                llm_vocab=self.llm_vocab,
                coder=self.coder,
                log=self.log,
            )
        elif value == "array":
            return BaseParameterGenerator(
                llm=self.llm,
                llm_vocab=self.llm_vocab,
                coder=self.coder,
                log=self.log,
            )
        elif value == "object":
            return BaseParameterGenerator(
                llm=self.llm,
                llm_vocab=self.llm_vocab,
                coder=self.coder,
                log=self.log,
            )
        else:
            return BaseParameterGenerator(
                llm=self.llm,
                llm_vocab=self.llm_vocab,
                coder=self.coder,
                log=self.log,
            )

    def _get_bool_paramgen(self) -> BoolParameterGenerator:
        if self.log is None:
            raise LogError(
                "Logger not defined, unable to pass logger to parameter\
generator."
            )
        return BoolParameterGenerator(
            function_names=["True", "False"],
            llm_vocab=self.llm_vocab,
            coder=self.coder,
            log=self.log,
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
        if self.log is None:
            raise LogError("Logger not found during output to JSON.")
        self.log.log(
            logging.INFO,
            f"Outputting responses to JSON to {self.path.args.output}...",
        )
        output_path = self.path.args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path.args.output, "w") as jfile:
            json.dump(self.output_list, jfile, indent=2)
        print(
            Color.GREEN.value
            + "\nOutput complete!\n"
            + Color.MAGENTA.value
            + "For more prompting, call me again, maybe?\n<End>\n"
        )

    def _print_input(self) -> None:
        print(
            Color.MAGENTA.value + "Initializing program..." + Color.RESET.value
        )
