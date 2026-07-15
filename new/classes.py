from pydantic import BaseModel, model_validator
from typing import Self


class Path(BaseModel):
    input: str = "data/input/function_calling_tests.json"
    output: str = "data/output/function_calls.json"
    func_def: str = "data/input/functions_definition.json"


class Vocab(BaseModel):
    vocab: dict[int, str]
    reverse: dict[str, int] = {}

    @model_validator(mode="after")
    def get_reverse(self) -> Self:
        self.reverse = {k: v for v, k in self.vocab.items()}
        return self

    def add_to_vocab(self, node: str) -> None:
        self.vocab[self.count_items()] = node

    def count_items(self) -> int:
        return len(self.vocab.items())


class TrieNode(BaseModel):
    vocab: Vocab
    children: list["TrieNode | None"]
    is_end_of_word: bool = False

    @model_validator(mode="after")
    def children(self) -> Self:
        self.children = [None] * len(self.vocab.vocab.items())
        return self


class Trie(BaseModel):
    root: TrieNode | None = None
    vocab: Vocab

    @model_validator(mode="after")
    def get_root(self) -> Self:
        self.root = TrieNode(vocab=self.vocab)
        return self

    # method to insert a key into the Trie
    def insert(self, key: str) -> None:
        current = self.root
        for char in key:
            index = self.vocab.reverse.get(char)
            if current.children[index] is None:
                current.children[index] = TrieNode(vocab=self.vocab)
            current = current.children[index]
        current.is_end_of_word = True

    # method to search a key in the trie
    def search(self, key: str) -> bool:
        current = self.root
        for char in key:
            index = self.vocab.reverse.get(char)
            if current.children[index] is None:
                return False
            current = current.children[index]
        return current.is_end_of_word

    # method to check if a prefix exists in the trie
    def is_prefix(self, prefix) -> bool:
        current = self.root
        for char in prefix:
            index = self.vocab.reverse.get(char)
            if current.children[index] is not None:
                current = current.children[index]
        return current != self.root


if __name__ == "__main__":
    vocab = {
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
    v = Vocab(vocab=vocab)
    tree = Trie(vocab=v)
    arr = ["fn_greet", "fn_hello_world", "fn_your_mom"]
    for f in arr:
        tree.insert(f)
    search = ["fn_yes", "fn_your_mom", "fn_no"]
    for f in search:
        if tree.search(f):
            print("true", end=" ")
        else:
            print("false", end=" ")
    print()
    prefix = ["haha_", "fn_he", "fn_n"]
    for s in prefix:
        if tree.is_prefix(s):
            print("true", end=" ")
        else:
            print("false", end=" ")
