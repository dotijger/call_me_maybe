from pydantic import BaseModel, model_validator
from src.error import TrieError
from typing import Self, TypedDict, Any
from enum import Enum


class Color(Enum):
    """ANSI escape codes used to color terminal output.

    Each member's value is the raw ANSI escape sequence that switches
    the terminal's foreground color (or, for ``RESET``, restores the
    default). Concatenate a member's ``value`` before text to color it,
    and append ``Color.RESET.value`` afterwards to stop the effect from
    bleeding into subsequent output.
    """

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
    """A bidirectional mapping between token ids and token strings.

    Wraps the LLM's vocabulary so that both directions of lookup
    (id -> token and token -> id) are available in O(1), which the
    constrained-decoding generators rely on at every generation step.

    Attributes:
        vocab (dict[int, str]): The vocabulary in id-to-token
            (decoding) direction.
        inverted (dict[str, int]): The vocabulary in token-to-id
            (encoding) direction. Computed automatically from
            ``vocab`` if not supplied.
        size (int | None): The number of entries in the vocabulary,
            computed automatically during validation.
    """

    vocab: dict[int, str]
    inverted: dict[str, int] = {}
    size: int | None = None

    @model_validator(mode="after")
    def init(self) -> Self:
        """Derives the inverted mapping and vocabulary size if needed.

        If ``inverted`` was not supplied, it is built by swapping the
        keys and values of ``vocab``. ``size`` is always (re)computed
        from ``vocab``.

        Returns:
            Self: The validated model instance, with ``inverted`` and
                ``size`` populated.
        """
        if len(self.inverted.items()) == 0:
            self.inverted = {k: v for v, k in self.vocab.items()}
        self.size = len(self.vocab.items())
        return self

    def add_to_vocab(self, node: str) -> None:
        """Appends a new token to the vocabulary at the next free id.

        Args:
            node (str): The token string to add.

        Returns:
            None
        """
        self.vocab[self.count_items()] = node

    def count_items(self) -> int:
        """Counts how many entries the vocabulary currently holds.

        Returns:
            int: The number of id-to-token entries in ``vocab``.
        """
        return len(self.vocab.items())


class TrieNode(BaseModel):
    """A single node in a character-level :class:`Trie`.

    Each node holds one child slot per character in the associated
    vocabulary, indexed by that character's vocabulary id, plus a flag
    marking whether the path from the root to this node spells out a
    complete entry.

    Attributes:
        vocab (Vocab): The character vocabulary defining how many
            child slots this node has and how characters map to child
            indices.
        children (list[TrieNode | None]): The node's children, indexed
            by character id; ``None`` where no child exists yet.
        is_end_of_word (bool): Whether the path ending at this node is
            a complete trie entry.
    """

    vocab: Vocab
    children: list["TrieNode | None"] = []
    is_end_of_word: bool = False

    @model_validator(mode="after")
    def setup_children(self) -> Self:
        """Allocates an empty child slot for every character in the vocab.

        Returns:
            Self: The validated model instance, with ``children``
                sized to match ``vocab``.
        """
        self.children = [None] * len(self.vocab.vocab.items())
        return self


class Trie(BaseModel):
    """A char-level trie used to constrain decoding to a fixed set of strings.

    Built once over a list of allowed entries (function names or
    boolean literals), the trie is then queried at every decoding step
    to determine whether a candidate token would keep the generated
    text a valid prefix of, or a complete match for, one of those
    entries.

    Attributes:
        root (TrieNode | None): The trie's root node, created during
            validation.
        vocab (Vocab): The restricted character vocabulary the trie is
            built over.
        entries (list[str]): The complete strings the trie should
            accept (e.g. function names or ``"True"``/``"False"``).
    """

    root: TrieNode | None = None
    vocab: Vocab
    entries: list[str]

    @model_validator(mode="after")
    def build_trie(self) -> Self:
        """Creates the root node and inserts every entry into the trie.

        Returns:
            Self: The validated model instance, with ``root`` built
                and populated from ``entries``.
        """
        self.root = TrieNode(vocab=self.vocab)
        for f in self.entries:
            self.insert(f)
        return self

    @property
    def trie_root(self) -> TrieNode:
        """Returns the trie's root node.

        Returns:
            TrieNode: The root of the trie.
        """
        assert self.root is not None
        return self.root

    # method to insert a key into the Trie
    def insert(self, key: str) -> None:
        """Inserts a complete string into the trie.

        Walks (and creates, as needed) one child node per character of
        ``key``, then marks the final node as the end of a word.

        Args:
            key (str): The string to insert.

        Returns:
            None

        Raises:
            TrieError: If a character in ``key`` is not present in the
                trie's vocabulary.
        """
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
        """Checks whether a string is a complete entry in the trie.

        Args:
            key (str): The string to look up.

        Returns:
            bool: ``True`` if ``key`` was inserted as a complete entry,
                ``False`` if it is missing or only a partial prefix.
        """
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
        """Checks whether a string is a valid prefix of some trie entry.

        Args:
            prefix (str): The candidate prefix to check.

        Returns:
            bool: ``True`` if there exists at least one full path in
                the trie starting with ``prefix``, ``False`` otherwise
                (including when ``prefix`` is empty or contains a
                character outside the trie's vocabulary).
        """
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
    """The raw shape of a single entry in ``functions_definition.json``.

    Attributes:
        name (str): The function's name.
        description (str): A natural-language description of what the
            function does.
        parameters (dict[str, dict[str, str]]): A mapping from
            parameter name to a small dict describing that parameter
            (at minimum, its ``"type"``).
        returns (dict[str, str]): A dict describing the function's
            return value (at minimum, its ``"type"``).
    """

    name: str
    description: str
    parameters: dict[str, dict[str, str]]
    returns: dict[str, str]


class OutputDict(TypedDict):
    """The shape of a single entry written to the output JSON file.

    Attributes:
        prompt (str): The original natural-language request.
        name (str): The name of the function selected to answer it.
        parameters (dict[str, Any]): The extracted, type-coerced
            arguments for that function.
    """

    prompt: str
    name: str
    parameters: dict[str, Any]
