#!/usr/bin/env python3

from src.functions_parser import schema_import, JSONraw
from src.parsing import import_prompts
from src.prompt import build_prompt
from llm_sdk import llm_sdk
import json


def llm() -> None:
    llm = llm_sdk.Small_LLM_Model()
    prompts = import_prompts()
    functions = schema_import("data/input/functions_definition.json")
    llm_prompts: list[str] = []
    for prompt in prompts:
        llm_prompts.append(build_prompt(prompt, functions))
    vocab_path = llm.get_path_to_vocab_file()
    with open(vocab_path) as v:
        raw_vocab = json.load(v)
    for prompt in llm_prompts:
        ids = llm.encode(prompt)
        print(ids.shape)
        ids = ids.squeeze(0).tolist()
        print(len(ids))
        logits = llm.get_logits_from_input_ids(ids)
        print(len(logits))


if __name__ == "__main__":
    llm()
