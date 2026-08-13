import numpy as np


def extract_outside(prompt: str, terminator: str) -> list[str]:
    """Extracts the whitespace-separated words that lie outside quotes.

    Walks through ``prompt`` character by character, toggling an
    "inside quotes" flag whenever ``terminator`` is encountered, and
    collects only the characters found outside of quoted spans.

    Args:
        prompt (str): The full text to scan.
        terminator (str): The quote character used to delimit quoted
            spans (e.g. ``'`` or ``"``).

    Returns:
        list[str]: The words found outside of any quoted span.
    """
    outsides = []
    inside = False
    current = ""
    for char in prompt:
        if char == terminator:
            inside = not inside
            outsides.append(current)
            current = ""
        elif not inside:
            current += char
    outsides.append(current)
    outside = " ".join(outsides)
    return outside.split()


def extract_substrings(prompt: str) -> list[str]:
    """Extracts both quoted substrings and unquoted words from a prompt.

    Determines the quote character to use (``'`` if the prompt contains
    an even, non-zero number of single quotes, otherwise ``"``), then
    returns every quoted span found together with the words that lie
    outside of quotes.

    Args:
        prompt (str): The natural language prompt to scan.

    Returns:
        list[str]: The quoted substrings followed by the words found
            outside of quotes.
    """
    terminator = (
        "'" if prompt.count("'") >= 2 and prompt.count("'") % 2 == 0 else '"'
    )
    substrings = []
    start = prompt.find(terminator)
    while start != -1:
        end = prompt.find(terminator, start + 1)
        if end == -1:
            break
        substrings.append(prompt[start+1:end])
        start = prompt.find(terminator, end + 1)
    outsides = extract_outside(prompt, terminator)
    return substrings + outsides


def extract_allowed_substrings(
    substrings: list[str], paramdict: dict[str, str]
) -> list[str]:
    """Filters out substrings that have already been used as parameters.

    Args:
        substrings (list[str]): Candidate substrings extracted from a
            prompt.
        paramdict (dict[str, str]): Parameters already extracted, keyed
            by parameter name.

    Returns:
        list[str]: The substrings that do not already appear among the
            values of ``paramdict``.
    """
    allowed = []
    for string in substrings:
        if string not in paramdict.values():
            allowed.append(string)
    return allowed


def replace_space(text: str) -> str:
    """Replaces literal spaces with the BPE pseudo-space character.

    Args:
        text (str): The text to transform.

    Returns:
        str: ``text`` with every space (`` ``) replaced by ``Ġ``, the
            byte-level BPE marker used to indicate a preceding space.
    """
    return text.replace(" ", "Ġ")


def replace_g(text: str) -> str:
    """Converts the BPE pseudo-space character back into literal spaces.

    If ``text`` starts with ``Ġ``, that leading marker is dropped (since
    it denotes a space that precedes the very first character rather
    than a space within the string) before the remaining occurrences
    are converted to spaces.

    Args:
        text (str): The text to transform, potentially containing
            ``Ġ`` markers.

    Returns:
        str: ``text`` with ``Ġ`` markers replaced by literal spaces.
    """
    if len(text) == 0:
        return ""
    if text[0] == "Ġ":
        return text[1:].replace("Ġ", " ")
    return text.replace("Ġ", " ")


def get_mask(
    logits: np.typing.ArrayLike, ids: list[int], non: list[int] | None
) -> np.typing.ArrayLike:
    """Builds an additive logit mask that restricts decoding to given ids.

    Every position in the returned mask is set to negative infinity,
    except for the positions listed in ``ids`` (minus any positions
    listed in ``non``), which are set to zero. Adding this mask to a
    logits array effectively zeroes out the probability of any
    disallowed token.

    Args:
        logits (np.typing.ArrayLike): The raw logits produced by the
            model, used only to determine the mask's length.
        ids (list[int]): The token ids that are allowed at this
            generation step.
        non (list[int] | None): Token ids to exclude from ``ids``
            (e.g. ids already tried and rejected). If ``None``, every
            id in ``ids`` is kept.

    Returns:
        np.typing.ArrayLike: An additive mask, the same length as
            ``logits``, with ``0`` at allowed positions and ``-inf``
            everywhere else.
    """
    # an id is also its 'index' in the vocabulary / the key
    logits_arr = np.asarray(logits)
    mask = np.full(len(logits_arr), -np.inf)
    ids_arr = np.asarray(ids, dtype=np.int64)

    if non is None:
        mask[ids_arr] = 0
        return mask

    non_arr = np.asarray(non, dtype=np.int64)
    keep = ~np.isin(ids_arr, non_arr)
    mask[ids_arr[keep]] = 0
    return mask


def get_substring(text: str) -> list[str]:
    """Extracts every quoted substring found in a text.

    Scans ``text`` for spans delimited by a single or double quote
    character and collects the content between each matching pair.

    Args:
        text (str): The text to scan for quoted spans.

    Returns:
        list[str]: The contents of each quoted span, in order of
            appearance.
    """
    substrings = []
    i = 0
    while i < len(text):
        char = text[i]
        if char in "'\"":
            end = text.find(char, i + 1)
            if end == -1:
                break
            substrings.append(text[i+1:end])
            i = end + 1
        else:
            i += 1
    return substrings


def is_prefix(small: str, big: str) -> bool:
    """Checks whether one string is a prefix of another.

    Args:
        small (str): The candidate prefix.
        big (str): The string to check ``small`` against.

    Returns:
        bool: ``True`` if ``small`` is a non-empty prefix of ``big``,
            ``False`` otherwise.
    """
    if len(small) > len(big) or len(small) == 0:
        return False
    for i in range(len(small)):
        if small[i] == big[i]:
            continue
        else:
            return False
    return True


def is_prefix_string(s: str, valid: dict[int, str]) -> bool:
    """Checks whether a string is a prefix of any value in a mapping.

    Args:
        s (str): The candidate prefix.
        valid (dict[int, str]): A mapping whose values are checked
            against ``s``.

    Returns:
        bool: ``True`` if ``s`` is a prefix of at least one value in
            ``valid``, ``False`` otherwise.
    """
    prefix = 0
    for value in valid.values():
        if is_prefix(s, value):
            prefix = 1
    return prefix == 1


def is_number(text: str) -> bool:
    """Checks whether a string can be parsed as a floating point number.

    Args:
        text (str): The text to test.

    Returns:
        bool: ``True`` if ``float(text)`` succeeds, ``False`` otherwise.
    """
    try:
        float(text)
        return True
    except ValueError:
        return False
