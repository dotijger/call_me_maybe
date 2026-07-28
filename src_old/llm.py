#!/usr/bin/env python3

from .functions_parser import schema_import, schema_creation
from .parsing import import_prompts
from .prompt import build_prompt
from llm_sdk import llm_sdk
import json


def llm() -> None:
    llm = llm_sdk.Small_LLM_Model()
    prompts = import_prompts()
    functions = schema_import("data/input/functions_definition.json")
    llm_prompts: list[str] = []
    for prompt in prompts:
        llm_prompts.append(build_prompt(prompt, functions))
    schemas = schema_creation(functions)
    vocab_path = llm.get_path_to_vocab_file()
    with open(vocab_path) as v:
        raw_vocab = json.load(v)
    vocab = {k: v for v, k in raw_vocab.items()}
    output_prompts = []
    for prompt in llm_prompts:
        output_prompts.append(json.loads(generate(prompt, schemas, vocab)))


def generate(
    prompt: str, schemas: list[list[dict[str, str]]], vocab: str
) -> str:
    ids = llm.encode(prompt).squeeze(0).tolist()
    prompt_name = get_prompt_name(prompt)
    generated = '{"prompt": ' + f'"{prompt_name}",'

    while not complete(schemas, generated):
        logits = llm.get_logits_from_input_ids(ids)
        allowed = allowed_tokens(schemas, generated, vocab)
    return generated + "}"


def get_prompt_name(prompt: str) -> str:
    start = prompt.find("user") + 14
    return prompt[start:]


def allowed_token(logits) -> list[int]:
    pass


def complete(schema: list[list[dict[str, str]]], text: str) -> bool:
    pass


if __name__ == "__main__":
    llm()
