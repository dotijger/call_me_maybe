#!/usr/bin/env python3


def is_prefix_string(s: str, valid: dict) -> bool:

    def is_prefix(small: str, big: str) -> bool:
        for i in range(len(small)):
            if small[i] == big[i]:
                continue
            else:
                return False
        return True

    prefix = 0
    for value in valid.values():
        if is_prefix(s, value):
            prefix = 1

    return prefix == 1

# prompt + name = add yourself to output
# ask llm for name of function (token) -> insert that into output
# fill 'parameters' into output and get valid parameters from dict
# ask llm for specific parameters (given prompt, fn name and specific parameters)
# add parameters one by one 

if __name__ == "__main__":
    d = {1: "hello", 2: "john", 3: "resonance"}
    print(is_prefix_string("res", d))
    print(is_prefix_string("jes", d))
