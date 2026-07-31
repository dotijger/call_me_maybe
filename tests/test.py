import json, sys
from llm_sdk.llm_sdk import Small_LLM_Model

# llm = Small_LLM_Model()
# vocab_path = llm.get_path_to_vocab_file()
# with open(vocab_path) as v:
# vocab = json.load(v)
# print(type(vocab))
# print(list(vocab.items())[:5])

test = "hellomynameisjasonhellothisnameisjason"
i = test.find("hellothis")
j = test.find("hellothis") + len("hellothis")
print(f"starts at {i}, next char at {j}")
for i, char in enumerate(test):
    print(f"index = {i}, char = {char}")


v = {
    0: "h",
    1: "hello",
    2: "hellothis",
    3: "jason",
    4: "name",
    5: "nameis",
    6: "my",
    7: "isjason",
    8: "this",
}


def find_match(string: str) -> int:
    id = -1
    for i in range(len(v.items())):
        if v.get(i) == string:
            id = i
    return id


def find_longest_match(ids: list[int]) -> int:
    max = 0
    longest = -1
    for id in ids:
        if len(v.get(id)) > max:
            max = len(v.get(id))
            longest = id
    return longest


def is_prefix(small: str, big: str) -> bool:
    for i in range(len(small)):
        if small[i] == big[i]:
            continue
        else:
            return False
    return True


def is_prefix_string(s: str, valid: dict) -> bool:
    prefix = 0
    for value in valid.values():
        if len(s) > len(value):
            continue
        if is_prefix(s, value):
            prefix = 1
    return prefix == 1


def encode(string: str) -> list[int]:
    ids = []
    possible_ids = []
    tokenized = ""
    i = 0
    sub = string[i]
    while tokenized != string:
        print(sub)
        tmp = find_match(sub)
        if tmp == -1:
            if tokenized + sub == string:
                break
            if is_prefix_string(sub, v):
                i += 1
                if i < len(string):
                    sub += string[i]
                continue
            if len(possible_ids) == 0:
                print("encodeerror")
                sys.exit(1)
            longest = find_longest_match(possible_ids)
            ids.append(longest)
            tokenized += v.get(longest)
            i = string.find(tokenized) + len(tokenized)
            possible_ids = []
            if i < len(string):
                sub = string[i]
        else:
            if tokenized + sub == string:
                possible_ids.append(tmp)
                break
            i += 1
            if i < len(string):
                sub += string[i]
            possible_ids.append(tmp)
    longest = find_longest_match(possible_ids)
    ids.append(longest)
    return ids


def decode(ids: list[int]) -> str:
    decoded = ""
    for id in ids:
        next_str = v.get(id)
        if next_str is None:
            print("error")
        decoded += next_str
    return decoded


print(test)
print(v)
encoded = encode(test)
print(encoded)
print(decode(encoded))
