from pydantic import BaseModel, model_validator, ConfigDict
from llm_sdk.llm_sdk import Small_LLM_Model
from src.error import VocabError
from src.classes import Vocab, Trie
from src.coder import Coder
import numpy as np
from src.helpers import (
    get_mask,
    replace_g,
    is_number,
    replace_space,
    extract_substrings,
)
from typing import Self


class BaseParameterGenerator(BaseModel):
    """
    gen =
    BaseParameterGenerator(llm=Small_LLM_Model,llm_vocab=Vocab,coder=Coder)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    llm: Small_LLM_Model
    llm_vocab: Vocab
    coder: Coder
    generated: str = ""

    def generate(
        self, input_ids: list[int], is_last: bool, context: str
    ) -> None:
        return None


class StringParameterGenerator(BaseParameterGenerator):
    """
    gen =
    StringParameterGenerator(llm=Small_LLM_Model,llm_vocab=Vocab,coder=Coder)
    """

    def generate(
        self, input_ids: list[int], is_last_parameter: bool, prompt: str
    ) -> str:
        self.generated = '"'
        generating = True
        terminator = "}" if is_last_parameter else ","
        end = f'"{terminator}'
        input_ids += self.coder.encode('"')
        substrings = extract_substrings(prompt)
        # print(f"allowed: {substrings=}")
        while generating is True:
            print(f"Generating: {self.generated!r}", flush=True)
            logits = np.array(self.llm.get_logits_from_input_ids(input_ids))
            allowed = self._allowed(is_last_parameter, substrings)
            if not allowed:
                generating = False
                break
            mask = get_mask(logits, allowed, None)
            masked = logits + mask
            next_id = np.argmax(masked)
            if self.generated.endswith(end) or len(self.generated) > 100:
                generating = False
            else:
                temp = self.llm_vocab.vocab.get(int(next_id))
                if temp is None:
                    raise VocabError(
                        "ID could not be found in the LLM vocab,\
                         generation aborting."
                    )
                self.generated += temp
                input_ids.append(int(next_id))
                content = replace_g(self.generated[1:])
                if self._completed(content, substrings):
                    generating = False
        self.generated = replace_g(self.generated)
        self.generated = self.generated.rstrip(end)
        self.generated = self.generated.lstrip('"')
        return self.generated

    def _allowed(
        self, is_last_parameter: bool, substrings: list[str]
    ) -> list[int]:
        allowed = []
        if self.generated == "":
            for value in self.llm_vocab.vocab.values():
                if self._valid_string_start(
                    value, is_last_parameter, substrings
                ):
                    id = self.llm_vocab.inverted.get(value)
                    if id is not None:
                        allowed.append(id)
        else:
            for value in self.llm_vocab.vocab.values():
                if self._valid_string_mid(
                    value, is_last_parameter, substrings
                ):
                    id = self.llm_vocab.inverted.get(value)
                    if id is not None:
                        allowed.append(id)
        return allowed

    def _valid_string_start(
        self,
        token: str,
        is_last_parameter: bool,
        substrings: list[str],
    ) -> bool:
        temp = self.generated + token
        if token == '"':
            return True
        if temp[0] != '"':
            return False
        if len(temp) > 1:
            return self._valid_string_mid(token, is_last_parameter, substrings)
        return temp == '"'

    def _valid_string_mid(
        self,
        token: str,
        is_last_parameter: bool,
        substrings: list[str],
    ) -> bool:
        temp = self.generated + token
        terminator = "}" if is_last_parameter else ","
        end = f'"{terminator}'
        if temp.endswith(end):
            length = len(end)
            content = replace_g(temp[1:-length])
            return self._completed(content, substrings)
        elif temp[1:].endswith('"'):
            content = replace_g(temp[1:-1])
            return self._completed(content, substrings)
        else:
            content = replace_g(temp[1:])
        return any(s.startswith(content) for s in substrings)

    @staticmethod
    def _completed(content: str, substrings: list[str]) -> bool:
        """
        Checks whether there is a longer string
        in substrings that we are looking for.
        """
        if content not in substrings:
            return False
        for s in substrings:
            if s != content and s.startswith(content):
                return False
        return True

    # return any(s.startswith(content) for s in allowed)


class RegexParameterGenerator(BaseParameterGenerator):
    """
    gen =
    RegexParameterGenerator(llm=Small_LLM_Model,llm_vocab=Vocab,coder=Coder)
    """

    is_pattern: bool = True
    REGEX_CHAR: str = "\\.^$*+?{}[]()|-"

    def generate(
        self,
        input_ids: list[int],
        is_last_parameter: bool,
        prompt: str,
    ) -> str:
        self.generated = '"'
        generating = True
        terminator = "}" if is_last_parameter else ","
        end = f'"{terminator}'
        input_ids += self.coder.encode('"')
        while generating is True:
            print(f"Generating: {self.generated!r}", flush=True)
            logits = np.array(self.llm.get_logits_from_input_ids(input_ids))
            allowed = self._allowed(is_last_parameter, prompt)
            if not allowed:
                generating = False
                break
            mask = get_mask(logits, allowed, None)
            masked = logits + mask
            next_id = np.argmax(masked)
            if self.generated.endswith(end):
                generating = False
            else:
                temp = self.llm_vocab.vocab.get(int(next_id))
                if temp is None:
                    raise VocabError(
                        "ID could not be found in the LLM vocab, \
                            generation aborting."
                    )
                self.generated += temp
                input_ids.append(int(next_id))
        self.generated = replace_g(self.generated)
        self.generated = self.generated.rstrip(end)
        self.generated = self.generated.lstrip('"')
        return self.generated

    def _allowed(self, is_last_parameter: bool, context: str) -> list[int]:
        allowed = []
        if self.generated == "":
            for value in self.llm_vocab.vocab.values():
                if self._valid_regex_start(value, is_last_parameter, context):
                    id = self.llm_vocab.inverted.get(value)
                    if id is not None:
                        allowed.append(id)
        else:
            for value in self.llm_vocab.vocab.values():
                if self._valid_regex_mid(value, is_last_parameter, context):
                    id = self.llm_vocab.inverted.get(value)
                    if id is not None:
                        allowed.append(id)
        return allowed

    def _valid_regex_start(
        self, token: str, is_last_parameter: bool, prompt: str
    ) -> bool:
        temp = self.generated + token
        if temp[0] != '"':
            return False
        if len(temp) > 1:
            return self._valid_regex_mid(token, is_last_parameter, prompt)
        return temp == '"'

    def _valid_regex_mid(
        self, token: str, is_last_parameter: bool, prompt: str
    ) -> bool:
        temp = self.generated + token
        terminator = "}" if is_last_parameter else ","
        end = f'"{terminator}'
        if temp.endswith(end):
            length = len(end)
            content = temp[1:-length]
        elif temp[1:].endswith('"'):
            content = temp[1:-1]
        else:
            content = temp[1:]
        return self._valid_content(content)

    def _valid_content(self, content: str) -> bool:
        if self.is_pattern:
            return all(c.isalnum() or c in self.REGEX_CHAR for c in content)
        if len(content) > 20:
            return False
        return all(c.isalnum() or c in " _-*" for c in content)


class IntegerParameterGenerator(BaseParameterGenerator):
    """
    gen =
    IntegerParameterGenerator(llm=Small_LLM_Model,llm_vocab=Vocab,coder=Coder)
    """

    def generate(
        self, input_ids: list[int], is_last_parameter: bool, prompt: str
    ) -> str:
        self.generated = ""
        generating = True
        terminator = " " if is_last_parameter else ","
        while generating is True:
            print(f"Generating: {self.generated!r}", flush=True)
            logits = np.array(self.llm.get_logits_from_input_ids(input_ids))
            allowed = self._allowed(is_last_parameter, prompt)
            if not allowed:
                generating = False
                break
            mask = get_mask(logits, allowed, None)
            masked = logits + mask
            next_id = np.argmax(masked)
            if self.generated.endswith(terminator) or len(self.generated) > 20:
                generating = False
            else:
                temp = self.llm_vocab.vocab.get(int(next_id))
                if temp is None:
                    raise VocabError(
                        "ID could not be found in the LLM vocab, \
                        generation aborting."
                    )
                self.generated += temp
                input_ids.append(int(next_id))
        self.generated = self.generated.rstrip(terminator)
        return self.generated

    def _allowed(self, is_last_parameter: bool, context: str) -> list[int]:
        allowed = []
        if self.generated == "":
            for value in self.llm_vocab.vocab.values():
                if self._valid_int_start(value, is_last_parameter, context):
                    id = self.llm_vocab.inverted.get(value)
                    if id is not None:
                        allowed.append(id)
        else:
            for value in self.llm_vocab.vocab.values():
                if self._valid_int_mid(value, is_last_parameter, context):
                    id = self.llm_vocab.inverted.get(value)
                    if id is not None:
                        allowed.append(id)
        return allowed

    def _valid_int_start(
        self, token: str, is_last_parameter: bool, prompt: str
    ) -> bool:
        temp = self.generated + token
        if len(temp) > 1:
            return self._valid_int_mid(token, is_last_parameter, prompt)
        if temp[0] == "-":
            return True
        if temp[0] == ".":
            return False
        return temp.isdigit() and temp in prompt

    def _valid_int_mid(
        self, token: str, is_last_parameter: bool, prompt: str
    ) -> bool:
        for char in token:
            if char == ".":
                return False
            if char == "-":
                return False
        temp = self.generated + token
        terminator = " " if is_last_parameter else ","
        if len(self.generated) >= 20:
            if token[-1] != terminator:
                return False
        if temp.endswith(terminator):
            nbr = temp[:-1]
        else:
            nbr = temp
        return is_number(nbr) and nbr in prompt


class NumberParameterGenerator(BaseParameterGenerator):
    """
    gen =
    NumberParameterGenerator(llm=Small_LLM_Model,llm_vocab=Vocab,coder=Coder)
    """

    def generate(
        self, input_ids: list[int], is_last_parameter: bool, prompt: str
    ) -> str:
        self.generated = ""
        generating = True
        terminator = " " if is_last_parameter else ","
        while generating is True:
            print(f"Generating: {self.generated!r}", flush=True)
            logits = np.array(self.llm.get_logits_from_input_ids(input_ids))
            allowed = self._allowed(is_last_parameter, prompt)
            if not allowed:
                generating = False
                break
            mask = get_mask(logits, allowed, None)
            masked = logits + mask
            next_id = np.argmax(masked)
            if self.generated.endswith(terminator) or len(self.generated) > 20:
                generating = False
            else:
                temp = self.llm_vocab.vocab.get(int(next_id))
                if temp is None:
                    raise VocabError(
                        "ID could not be found in the LLM vocab, \
                        generation aborting."
                    )
                self.generated += temp
                input_ids.append(int(next_id))
        self.generated = self.generated.rstrip(terminator)
        return self.generated

    def _allowed(self, is_last_parameter: bool, context: str) -> list[int]:
        allowed = []
        if self.generated == "":
            for value in self.llm_vocab.vocab.values():
                if self._valid_flt_start(value, is_last_parameter, context):
                    id = self.llm_vocab.inverted.get(value)
                    if id is not None:
                        allowed.append(id)
        else:
            if self._check_dots(self.generated):
                for value in self.llm_vocab.vocab.values():
                    if self._valid_flt_mid_dot(
                        value, is_last_parameter, context
                    ):
                        id = self.llm_vocab.inverted.get(value)
                        if id is not None:
                            allowed.append(id)
            else:
                for value in self.llm_vocab.vocab.values():
                    if self._valid_flt_mid_no_dot(
                        value, is_last_parameter, context
                    ):
                        id = self.llm_vocab.inverted.get(value)
                        if id is not None:
                            allowed.append(id)
        return allowed

    def _valid_flt_start(
        self, token: str, is_last_parameter: bool, prompt: str
    ) -> bool:
        temp = self.generated + token
        if len(temp) > 1:
            return self._valid_flt_mid_no_dot(token, is_last_parameter, prompt)
        if temp[0] == "-" or temp[0] == ".":
            return True
        return temp.isdigit() and temp in prompt

    def _valid_flt_mid_no_dot(
        self, token: str, is_last_parameter: bool, prompt: str
    ) -> bool:
        dots = 0
        for char in token:
            if char == "-":
                return False
            if char == ".":
                dots += 1
        if dots > 1:
            return False
        temp = self.generated + token
        terminator = " " if is_last_parameter else ","
        if len(self.generated) >= 20:
            if token[-1] != terminator:
                return False
        if temp.endswith(terminator):
            nbr = temp[:-1]
        else:
            nbr = temp
        return is_number(nbr) and nbr in prompt

    def _valid_flt_mid_dot(
        self, token: str, is_last_parameter: bool, prompt: str
    ) -> bool:
        for char in token:
            if char == ".":
                return False
            if char == "-":
                return False
        temp = self.generated + token
        terminator = " " if is_last_parameter else ","
        if len(self.generated) >= 20:
            if token[-1] != terminator:
                return False
        if temp.endswith(terminator):
            nbr = temp[:-1]
        else:
            nbr = temp
        return is_number(nbr) and nbr in prompt

    @staticmethod
    def _check_dots(text: str) -> bool:
        dots = 0
        for char in text:
            if char == ".":
                dots += 1
        return dots != 0


class BoolParameterGenerator(BaseModel):
    """
    ng =
    BooleanGenerator(
    function_names=list[str],
    llm=Small_LLM_Model(),
    llm_vocab=Vocab(),coder=Coder
    )
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    function_names: list[str]
    llm_vocab: Vocab
    coder: Coder
    trie_vocab: Vocab = Vocab(
        vocab={
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
            31: "T",
            32: "F",
        }
    )
    trie_functions: Trie | None = None

    @model_validator(mode="after")
    def load(self) -> Self:
        if self.trie_functions is None:
            self.trie_functions = Trie(
                vocab=self.trie_vocab, entries=self.function_names
            )
        return self

    @property
    def trie_entries(self) -> Trie:
        assert self.trie_functions is not None
        return self.trie_functions

    def _allowed(self, generated: str) -> list[int]:
        allowed = []
        for value in self.llm_vocab.vocab.values():
            if self.trie_entries.is_prefix(generated + value):
                id = self.llm_vocab.inverted.get(value)
                if id is not None:
                    allowed.append(id)
            elif self.trie_entries.search(generated + value):
                id = self.llm_vocab.inverted.get(value)
                if id is not None:
                    allowed.append(id)
            else:
                continue
        return allowed

    def generate(
        self, llm: Small_LLM_Model, llm_prompt: list[int], prompt: str
    ) -> str:
        """
        paramgen.generate(
        input_ids, last_parameter, "function specific prompt"
        )
        """
        generated = ""
        generating = True
        input_ids = []
        not_allowed: list[int] = []
        prompts = f"Answer this prompt: {prompt}"
        text = replace_space(prompts)
        input_ids = llm_prompt + self.coder.encode(text)
        while generating is True:
            print(f"Generating: {generated!r}", flush=True)
            logits = np.array(llm.get_logits_from_input_ids(input_ids))
            allowed = self._allowed(generated)
            mask = get_mask(logits, allowed, not_allowed)
            masked = logits + mask
            next_id = np.argmax(masked)
            if self.trie_entries.search(generated):
                generating = False
            else:
                temp = self.llm_vocab.vocab.get(int(next_id))
                if temp is None:
                    raise VocabError(
                        "ID could not be found in the LLM vocab, \
                        generation aborting."
                    )
                temp_gen = generated + temp
                if self.trie_entries.is_prefix(temp_gen):
                    not_allowed = []
                    input_ids.append(int(next_id))
                    generated += temp
                else:
                    not_allowed.append(int(next_id))
        return generated
