class TrieNode:
    def __init__(self) -> None:
        self.children = [None]
        self.isEndOfWord = False


vocab = {
    1: "{",
    2: "}",
    3: ",",
    4: ":",
    5: "name: ",
    6: "parameters: ",
    7: "c",
}


class Trie:
    def __init__(self) -> None:
        self._root = TrieNode()

    # method to insert a key into the Trie
    def insert(self, key) -> None:
        current = self._root
        for char in key:
            ...

    # method to search a key in the trie
    def search(self, key) -> bool:
        return True

    # method to check if a prefix exists in the trie
    def is_prefix(self, prefix) -> bool:
        return True
