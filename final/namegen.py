from pydantic import BaseModel, model_validator, ConfigDict
from final.classes import Trie, Vocab
from llm_sdk.llm_sdk import Small_LLM_Model
from final.helpers import replace_space, get_mask
from final.coder import Coder
from typing import Self
import numpy as np


class NameGenerator(BaseModel):
    """
    ng = NameGenerator(function_names=list[str],llm_vocab=Vocab(),llm_prompt=str,coder=Coder)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    function_names: list[str]
    llm_vocab: Vocab
    llm_prompt: str
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
        }
    )
    trie_functions: Trie | None = None

    @model_validator(mode="after")
    def load(self) -> Self:
        if self.trie_functions is None:
            self.trie_functions = Trie(
                vocab=self.trie_vocab, entries=self.function_names
            )
        return Self

    def generate(self, llm: Small_LLM_Model, prompt: str) -> str:
        generated = ""
        generating = True
        input_ids = []
        not_allowed = []
        prompts = f"{self.llm_prompt}. Answer this prompt: {prompt}"
        text = replace_space(prompts)
        input_ids += self.coder.encode(text)
        print(repr(self.coder.decode(input_ids)))
        print(type(input_ids))
        while generating is True:
            print(f"step, generated so far: {generated!r}", flush=True)
            logits = np.array(llm.get_logits_from_input_ids(input_ids))
            allowed = self._allowed(generated)
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

        def _allowed(self, generated: str) -> list[int]:
            allowed = []
            for value in self.llm_vocab.vocab.values():
                if self.trie_functions.is_prefix(generated + value):
                    allowed.append(self.llm_vocab.inverted.get(value))
                elif self.trie_functions.search(generated + value):
                    allowed.append(self.llm_vocab.inverted.get(value))
                else:
                    continue
            return allowed
