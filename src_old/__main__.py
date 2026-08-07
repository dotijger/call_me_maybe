from src.constraineddecoder import ConstrainedDecoder
from src.error import (
    EncodeError,
    DecodeError,
    ParameterError,
    ParsingError,
    VocabError,
    TrieError,
)
from llm_sdk.llm_sdk import Small_LLM_Model
from datetime import datetime


if __name__ == "__main__":
    ai = Small_LLM_Model()
    start = datetime.now()
    try:
        cd = ConstrainedDecoder(llm=ai)
        cd.run()
    except (
        EncodeError,
        DecodeError,
        ParameterError,
        ParsingError,
        VocabError,
        TrieError,
    ) as e:
        print(e)
    end = datetime.now()
    duration = end - start
    print(f"Duration of the program: {duration.total_seconds():.3f}s")
    # try:
    # cd.run()
    # except Exception as e:
    # print(e)
    # sys.exit(1)
