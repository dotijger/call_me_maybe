from pydantic import BaseModel
from constrained import is_prefix_string


class Path(BaseModel):
    input: str = "data/input/function_calling_tests.json"
    output: str = "data/output/function_calls.json"
    func_def: str = "data/input/functions_definition.json"


class Vocab(BaseModel):
    vocab: dict[int, str]
    reverse: dict[str, int]

    def __init__(self) -> None:
        reverse = {k, v for v, k in vocab.items()}

    def add_to_vocab(self, node: str) -> None:
        vocab[self.count_items()] = node

    def count_items(self) -> int:
        return len(vocab.items())


class TrieNode(BaseModel):
    vocab: Vocab
    children: list[None]
    is_end_of_word: bool = False

    def __init__(self) -> None:
        self.children = [None] * len(self.vocab.vocab.items())


class Trie(BaseModel):
    root: TrieNode
    vocab: Vocab

    # method to insert a key into the Trie
    def insert(self, key: str) -> None:
        current = self.root
        inserted = ""
        for char in key:
            if is_prefix_string(inserted, self.vocab.vocab):
                new_node = TrieNode(vocab=self.vocab)
                index = self.vocab.reverse.get(inserted)
                current.children[index] = new_node
                current = current.children[index]
                inserted = ""
            inserted += char
        if inserted != "":
            raise TrieError("Unable to import full key, vocabulary insufficient")


    # method to search a key in the trie
    def search(self, key: str) -> bool:
        current = self.root
        while current.children[vocab.reverse[key]] == None:
            for child in current.children:
                if child != None:
                    current = child
        return True

    # method to check if a prefix exists in the trie
    def is_prefix(self, prefix) -> bool:
        return True


vocab = {
    1: "{",
    2: "}",
    3: ",",
    4: ":",
    5: "name: ",
    6: "parameters: ",
    7: "c",
}
