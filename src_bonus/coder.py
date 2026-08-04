from pydantic import BaseModel
from src.classes import Vocab, Color
from src.error import DecodeError, EncodeError
from src.helpers import is_prefix_string


class Coder(BaseModel):
    llm_vocab: Vocab

    def decode(self, ids: list[int]) -> str:
        decoded = ""
        for id in ids:
            next_str = self.llm_vocab.vocab.get(id)
            if next_str is None:
                raise DecodeError("Given ID is not in LLM vocab")
            decoded += next_str
        return decoded

    def encode(self, string: str) -> list[int]:
        ids = []
        possible_ids: list[int] = []
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
                if is_prefix_string(sub, self.llm_vocab.vocab):
                    i += 1
                    if i < len(string):
                        sub += string[i]
                        continue
                if len(possible_ids) == 0:
                    # print(tokenized, string)
                    print(
                        Color.RED.value
                        + "\nSpecial character detected, \
falling back to LLM encoder.\n"
                        + Color.RESET.value
                    )
                    raise EncodeError(
                        "String cannot be encoded, vocabulary insufficient"
                    )
                longest = self._find_longest_match(possible_ids)
                ids.append(longest)
                possible_ids = []
                token = self.llm_vocab.vocab.get(longest)
                if token is not None:
                    tokenized += token
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
        id = self.llm_vocab.inverted.get(string, -1)
        return id

    def _find_longest_match(self, ids: list[int]) -> int:
        max = 0
        longest = -1
        for id in ids:
            token = self.llm_vocab.vocab.get(id)
            if token is not None:
                if len(token) > max:
                    max = len(token)
                    longest = id
        return longest
