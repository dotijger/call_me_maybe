from llm_sdk.llm_sdk import Small_LLM_Model
import json


if __name__ == "__main__":
    llm = Small_LLM_Model()
    path = llm.get_path_to_vocab_file()
    with open(path) as f:
        vocab = json.load(f)
    for token, id in vocab.items():
        if "is" in token and len(token) <= 4:
            print(repr(token), id)
