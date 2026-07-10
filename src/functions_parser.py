#!/usr/bin/env python3

from pathlib import Path


def schema_import(file: str) -> None:
    with open(file, "r") as file:
        raw = file.read()
        raw_split = list(raw.split("name"))
        for functions in raw_split:
            print("next function:")
            print(functions)


# experiemtn with parsing per function to get a short description to give in the prompt to the LLM
# make it a list with per function a list (so give the list of lists printed to LLM along with the prompt from the user)
# output always is the prompt (easy) + the schema_walker per selection fucntion (so allowed schemas for each function)
# allowed schemas are created here

if __name__ == "__main__":
    HERE = Path(__file__).parent
    schema_import(HERE / "functions_definitions.json")
