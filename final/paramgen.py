from pydantic import BaseModel, model_validator, ConfigDict
from llm_sdk.llm_sdk import Small_LLM_Model
from final.classes import Vocab
from final.coder import Coder
from final.error import NoParameterDefinedError


class BaseParameterGenerator(BaseModel):
    """
    gen = BaseParameterGenerator(llm=Small_LLM_Model,llm_vocab=Vocab,coder=Coder)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    llm: Small_LLM_Model
    llm_vocab: Vocab
    coder: Coder

    def generate(self, input_ids: list[int], is_last: bool, context: str) -> str:
        raise NoParameterDefinedError(
            "Calling generation on a base generator class instance."
        )


class StringParameterGenerator(BaseParameterGenerator):
    """
    gen = StringParameterGenerator(llm=Small_LLM_Model,llm_vocab=Vocab,coder=Coder)
    """

    def generate(self, input_ids: list[int], is_last: bool, context: str) -> str:
        pass


class RegexParameterGenerator(BaseParameterGenerator):
    """
    gen = RegexParameterGenerator(llm=Small_LLM_Model,llm_vocab=Vocab,coder=Coder)
    """

    def generate(self, input_ids: list[int], is_last: bool, context: str) -> str:
        pass


class IntegerParameterGenerator(BaseParameterGenerator):
    """
    gen = IntegerParameterGenerator(llm=Small_LLM_Model,llm_vocab=Vocab,coder=Coder)
    """

    def generate(self, input_ids: list[int], is_last: bool, context: str) -> str:
        pass


class NumberParameterGenerator(BaseParameterGenerator):
    """
    gen = NumberParameterGenerator(llm=Small_LLM_Model,llm_vocab=Vocab,coder=Coder)
    """

    def generate(self, input_ids: list[int], is_last: bool, context: str) -> str:
        pass
