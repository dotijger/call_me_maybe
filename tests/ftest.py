from final.constraineddecoder import ConstrainedDecoder
from llm_sdk.llm_sdk import Small_LLM_Model
import sys


if __name__ == "__main__":
    ai = Small_LLM_Model()
    cd = ConstrainedDecoder(llm=ai)
    try:
        cd.run()
    except Exception as e:
        print(e)
        sys.exit(1)
