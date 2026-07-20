import json
from llm_sdk.llm_sdk import Small_LLM_Model

llm = Small_LLM_Model()
vocab_path = llm.get_path_to_vocab_file()
with open(vocab_path) as v:
    vocab = json.load(v)
print(type(vocab))
print(list(vocab.items())[:5])
