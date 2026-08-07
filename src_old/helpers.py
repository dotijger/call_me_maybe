import numpy as np


def extract_outside(prompt: str, terminator: str) -> list[str]:
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
    allowed = []
    for string in substrings:
        if string not in paramdict.values():
            allowed.append(string)
    return allowed


def replace_space(text: str) -> str:
    return text.replace(" ", "Ġ")


def replace_g(text: str) -> str:
    if len(text) == 0:
        return ""
    if text[0] == "Ġ":
        return text[1:].replace("Ġ", " ")
    return text.replace("Ġ", " ")


def get_mask(
    logits: np.typing.ArrayLike, ids: list[int], non: list[int] | None
) -> np.typing.ArrayLike:
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
    if len(small) > len(big) or len(small) == 0:
        return False
    for i in range(len(small)):
        if small[i] == big[i]:
            continue
        else:
            return False
    return True


def is_prefix_string(s: str, valid: dict[int, str]) -> bool:
    prefix = 0
    for value in valid.values():
        if is_prefix(s, value):
            prefix = 1
    return prefix == 1


def is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False
