import numpy as np


def replace_space(text: str) -> str:
    return text.replace(" ", "Ġ")


def replace_g(text: str) -> str:
    if text[0] == "Ġ":
        return text[1:].replace("Ġ", " ")
    return text.replace("Ġ", " ")


def get_mask(
    logits: list[float], ids: list[int], non: list[int] | None
) -> np.typing.ArrayLike:
    # an id is also its 'index' in the vocabulary / the key
    mask = np.full(len(logits), -np.inf)
    if non is None:
        for id in ids:
            mask[id] = 0
        return mask
    for id in ids:
        if id not in non:
            mask[id] = 0
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
            substrings.append(text[i + 1 : end])
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


def is_prefix_string(self, s: str, valid: dict) -> bool:
    prefix = 0
    for value in valid.values():
        if self._is_prefix(s, value):
            prefix = 1
    return prefix == 1
