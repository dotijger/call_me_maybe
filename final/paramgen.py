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

    # parameter prompt generator
    def prompt(self, info: tuple[str, int], count: int, prompt: str, kind: str) -> str:
        parameters = self.param_schema[info[0]]
        param_to_extract = parameters[count - 1][0]
        prompt = f"User prompt: {prompt} \
        Function being called: {info[0]} \
        Parameter to extract: {param_to_extract}. \
        Extract the value of {param_to_extract} from the user prompt. \
        Copy the {kind} parameter from the prompt: \
        {prompt}. {param_to_extract} = "
        return prompt

    def generate(self, input_ids: list[int], is_last: bool, context: str) -> str:
        raise NoParameterDefinedError(
            "Calling generation on a base generator class instance."
        )


class StringParameterGenerator(BaseParameterGenerator):
    """
    gen = StringParameterGenerator(llm=Small_LLM_Model,llm_vocab=Vocab,coder=Coder)
    """

    def _get_string(
        self,
        input_ids: list[int],
        is_last_parameter: bool,
        prompt: str,
    ) -> str:
        generated = '"'
        generating = True
        terminator = "}" if is_last_parameter else ","
        end = f'"{terminator}'
        input_ids += self.coder.encode('"')
        while generating is True:
            print(f"step, generated so far: {generated!r}", flush=True)
            logits = np.array(self.llm.get_logits_from_input_ids(input_ids))
            allowed = self._allowed(generated, "string", is_last_parameter, "regex")
            if not allowed:
                generating = False
                break
            mask = get_mask(logits, allowed, None)
            masked = logits + mask
            next_id = np.argmax(masked)
            if generated.endswith(end):
                generating = False
            else:
                generated += self.llm_vocab.vocab.get(next_id)
                input_ids.append(next_id)
        generated = replace_g(generated)
        return generated

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

    def _get_number(
        self, input_ids: list[int], is_last_parameter: bool, prompt: str
    ) -> str:
        generated = ""
        generating = True
        terminator = " " if is_last_parameter else ","
        while generating is True:
            print(f"step, generated so far: {generated!r}", flush=True)
            logits = np.array(self.llm.get_logits_from_input_ids(input_ids))
            allowed = self._allowed(generated, "number", is_last_parameter, prompt)
            if not allowed:
                generating = False
                break
            mask = get_mask(logits, allowed, None)
            masked = logits + mask
            next_id = np.argmax(masked)
            if generated.endswith(terminator) or len(generated) > 20:
                generating = False
            else:
                generated += self.llm_vocab.vocab.get(next_id)
                input_ids.append(next_id)
        if is_last_parameter:
            generated = generated.rstrip(" ") + "}"
        return generated

    def generate(self, input_ids: list[int], is_last: bool, context: str) -> str:
        pass
