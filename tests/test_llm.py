from new.setup import ConstrainedDecoder
from new.error import EncodeError, DecodeError
from llm_sdk.llm_sdk import Small_LLM_Model

if __name__ == "__main__":
    decoder = ConstrainedDecoder(llm=Small_LLM_Model())
    try:
        decoder.run()
    except (EncodeError, DecodeError) as e:
        print(e)
