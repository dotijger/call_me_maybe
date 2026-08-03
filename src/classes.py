from pydantic import BaseModel, model_validator
from src.error import TrieError
from typing import Self, TypedDict, Any
from enum import Enum


class Color(Enum):
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    LIGHT_GRAY = "\033[37m"
    DARK_GRAY = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    WHITE = "\033[97m"
    RESET = "\033[0m"


class Vocab(BaseModel):
    vocab: dict[int, str]
    inverted: dict[str, int] = {}
    size: int | None = None

    @model_validator(mode="after")
    def init(self) -> Self:
        if len(self.inverted.items()) == 0:
            self.inverted = {k: v for v, k in self.vocab.items()}
        self.size = len(self.vocab.items())
        return self

    def add_to_vocab(self, node: str) -> None:
        self.vocab[self.count_items()] = node

    def count_items(self) -> int:
        return len(self.vocab.items())


class TrieNode(BaseModel):
    vocab: Vocab
    children: list["TrieNode | None"] = []
    is_end_of_word: bool = False

    @model_validator(mode="after")
    def setup_children(self) -> Self:
        self.children = [None] * len(self.vocab.vocab.items())
        return self


class Trie(BaseModel):
    root: TrieNode | None = None
    vocab: Vocab
    entries: list[str]

    @model_validator(mode="after")
    def build_trie(self) -> Self:
        self.root = TrieNode(vocab=self.vocab)
        for f in self.entries:
            self.insert(f)
        return self

    @property
    def trie_root(self) -> TrieNode:
        assert self.root is not None
        return self.root

    # method to insert a key into the Trie
    def insert(self, key: str) -> None:
        if self.root is not None:
            current = self.trie_root
            for char in key:
                index = self.vocab.inverted.get(char)
                if index is None:
                    raise TrieError(
                        f"TrieVocab insufficient for \
                        string {key}, {char} not found."
                    )
                next = current.children[index]
                if next is None:
                    next = TrieNode(vocab=self.vocab)
                    current.children[index] = next
                current = next
            current.is_end_of_word = True

    # method to search a key in the trie
    def search(self, key: str) -> bool:
        current = self.trie_root
        for char in key:
            index = self.vocab.inverted.get(char)
            if index is None:
                return False
            next = current.children[index]
            if next is None:
                return False
            current = next
        return current.is_end_of_word

    # method to check if a prefix exists in the trie
    def is_prefix(self, prefix: str) -> bool:
        current = self.trie_root
        i = 0
        for char in prefix:
            index = self.vocab.inverted.get(char)
            if index is None:
                return False
            next = current.children[index]
            if next is not None:
                current = next
            else:
                i += 1
        if i > 0:
            return False
        return current != self.root


class JSONraw(TypedDict):
    name: str
    description: str
    parameters: dict[str, dict[str, str]]
    returns: dict[str, str]


class OutputDict(TypedDict):
    prompt: str
    name: str
    parameters: dict[str, Any]
