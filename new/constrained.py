#!/usr/bin/env python3

from pydantic import BaseModel


class ConstrainedDecoder(BaseModel):
    prompts: list[str]
    functions: list[str]
    parameters: list[dict[str,]]


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
        if is_prefix(s, value):
            prefix = 1

    return prefix == 1


def main() -> None:
    pass
    # parse input -> store input in path
    # get prompts from input / path


#   start output json:
#   opening brackets
#       for each prompt:
#       1. hardcode prompt up until name
#       2. prompt llm for which function name (give it list of function names and the prompt)
#       3. fill in function name (from llm)
#       4. hardcode up until parameters, and get which parameters to ask for given function name
#           - need a dictionary to store the parameters per function name
#       5. prompt llm for each parameter (loop until all parameters retrieved)
#       6. hardcode parameters
#   closing brackets
#   return


# prompt + name = add yourself to output
# ask llm for name of function (token) -> insert that into output
# fill 'parameters' into output and get valid parameters from dict
# ask llm for specific parameters (given prompt, fn name and specific parameters)
# add parameters one by one

if __name__ == "__main__":
    main()
