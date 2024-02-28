from __future__ import annotations

from enum import Enum
from typing import List, Optional
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mgz.models.nlp.base_transformer import DecoderTransformer
GPT_TURN = f"<|end_of_turn|>GPT4 Correct Assistant:"


class PromptType(Enum):
    MISTRAL = 0
    ADAPT = 1


class PromptConfig:
    @classmethod
    def infer_prompt_type(cls, model: DecoderTransformer):
        if hasattr(model.config, '_name_or_path'):
            assert isinstance(model.config._name_or_path, str)
            if "adapt" in model.config._name_or_path.lower() or "instruct" in model.config._name_or_path.lower():
                return PromptType.ADAPT
            elif "openchat" in model.config._name_or_path.lower() or "mistral" in model.config._name_or_path.lower():
                return PromptType.MISTRAL
            else:
                raise ValueError(
                    "Could not determine prompt type, pass it as an argument")
        else:
            raise ValueError(
                "Could not determine prompt type, pass it as an argument")

    def __init__(self, system_context: Optional[str] = None,
                 model: DecoderTransformer = None,
                 prompt_type: PromptType = None,
                 question_wording_component: Optional[str] = None,):
        if prompt_type is None:
            self.prompt_type = self.infer_prompt_type(model)
        else:
            self.prompt_type = prompt_type
        self.system_context = system_context
        self.truncate_token_start = '[TRUNCABTABLE_START]'
        self.truncate_token_end = '[TRUNCABTABLE_END]'
        self.question_wording_component = question_wording_component


class PromptingInput:
    @classmethod
    def from_list(cls,
                  prompt_config: PromptConfig,
                  document_texts: List[str],
                  document_requests_list: Optional[
                      List[str] | List[List[str]]] = None,
                  document_type: str = "e-mail"):
        return [cls(prompt_config=prompt_config,
                    document_text=document_text,
                    document_requests=document_requests,
                    document_type=document_type)
                for document_text, document_requests in
                zip(document_texts, document_requests_list)]

    @classmethod
    def from_inferred_prompt_type(cls, model: DecoderTransformer,
                                  document_text: str,
                                  document_requests: Optional[
                                      str | List[str]] = None,
                                  document_type: str = "e-mail",
                                  system_context: str = ""):
        return cls(prompt_config=PromptConfig(model=model,
                                              system_context=system_context),
                   document_text=document_text,
                   document_requests=document_requests,
                   document_type=document_type)

    @staticmethod
    def prompt_mistral(main_body: str,
                       system_context: Optional[str] = None,
                       additional_chat: List[str] = []):
        """
        <s>[INST] <<SYS>>
        {{ system_prompt }}
        <</SYS>>

        {{ user_msg_1 }} [/INST] {{ model_answer_1 }} </s><s>[INST] {{ user_msg_2 }} [/INST]

        """

        if system_context:
            prompt = f"GTP4 Correct User: " \
                     f"{system_context}\n"
        else:
            prompt = f"GTP4 Correct User: "
        prompt += f"{main_body}\n" \
                  f"<|end_of_turn|>" \
                  f"GPT4 Correct Assistant:"
        for chat in additional_chat:
            prompt += f"<|end_of_turn|>" \
                      f"GPT4 Correct User: " \
                      f"{chat}\n" \
                      f"GPT4 Correct Assistant:"
        return prompt

    @staticmethod
    def prompt_adapt(main_body: str,
                     system_context: Optional[str] = None,
                     additional_chat: List[str] = []):
        """
        <s>[INST] <<SYS>>
        {{ system_prompt }}
        <</SYS>>

        {{ user_msg_1 }} [/INST] {{ model_answer_1 }} </s><s>[INST] {{ user_msg_2 }} [/INST]

        """
        if system_context:
            prompt = f"<s>[INST] <<SYS>>{system_context}<</SYS>>\n\n{main_body} [/INST]"
        else:
            prompt = f"<s>[INST] \n\n{main_body} [/INST]"
        for chat in additional_chat:
            prompt += f" </s><s>[INST]" \
                      f"{chat} [/INST]"
        return prompt

    def __init__(self,
                 prompt_config: PromptConfig,
                 document_text: str,
                 document_requests: Optional[str | List[str]] = None,
                 document_type: str = "e-mail",
                 ):
        self.document_text = document_text
        self.document_requests = document_requests
        self.document_type = document_type
        self.prompt_config = prompt_config
        self.prompt_type = prompt_config.prompt_type

        self.truncate_token_start = prompt_config.truncate_token_start
        self.truncate_token_end = prompt_config.truncate_token_end
        self.system_context = prompt_config.system_context

    def get_tokenizer_input(self, add_trunc: bool = True):
        raise NotImplementedError(
            "This method should be implemented in the child class")


class ContextPromptingInput(PromptingInput):

    def get_tokenizer_input(self, add_trunc: bool = True):

        def _make_tag_prompt(tag: str) -> str:
            # return f"We are looking for {tag}. Is this {self.document_type} what we are looking for. Yes or no.\n"
            # return f"Is it apparent that the {self.document_type} is part of \"{tag}\"? Yes or no.\n"
            return f"Is the {self.document_type} part of \"{tag}\"? Answer yes or no"

        if isinstance(self.document_requests, str):
            if add_trunc:
                prompt_body = f"{self.truncate_token_start}{self.document_text}{self.truncate_token_end}\n{_make_tag_prompt(self.document_requests)}"
            else:
                prompt_body = f" {self.document_text} \n{_make_tag_prompt(self.document_requests)}"
            if self.prompt_type == PromptType.MISTRAL:
                return self.prompt_mistral(prompt_body,
                                           system_context=self.system_context)
            else:
                return self.prompt_adapt(prompt_body,
                                         system_context=self.system_context)
        else:
            if add_trunc:
                prompt_body = f"{self.truncate_token_start} {self.document_text} {self.truncate_token_end}\n{_make_tag_prompt(self.document_requests[0])}"
            else:
                prompt_body = f" {self.document_text} \n{_make_tag_prompt(self.document_requests[0])}"
            additional_tag_prompts: List[str] = [
                _make_tag_prompt(document_request)
                for document_request in
                self.document_requests[1:]]
            if self.prompt_type == PromptType.MISTRAL:
                return self.prompt_mistral(main_body=prompt_body,
                                           additional_chat=additional_tag_prompts,
                                           system_context=self.system_context)
            else:
                # TODO 20 and 2 * are both random estimates
                return self.prompt_adapt(main_body=prompt_body,
                                         additional_chat=additional_tag_prompts,
                                         system_context=self.system_context)


class SummarizePromptInput(PromptingInput):
    @classmethod
    def from_list(cls,
                  prompt_config: PromptConfig,
                  document_texts: List[str],
                  word_limit: int = 50,
                  document_type: str = "e-mail"):
        return [cls(prompt_config=prompt_config,
                    document_text=document_text,
                    word_limit=word_limit,
                    document_type=document_type)
                for document_text in document_texts]

    def __init__(self, prompt_config: PromptConfig,
                 document_text: str,
                 document_type: str = "e-mail",
                 word_limit: int = 50,
                 ):
        super().__init__(
            prompt_config=prompt_config,
            document_text=document_text,
            document_type=document_type,
        )
        self.word_limit = word_limit

    def get_tokenizer_input(self, add_trunc: bool = True):
        if self.prompt_config.question_wording_component is not None:
            question = self.prompt_config.question_wording_component
        else:
            question = "Could you summarize this in at most"
        if add_trunc:
            qa_prefix = f"{question} {self.word_limit} words?\n{self.truncate_token_start}{self.document_text}{self.truncate_token_end}"
        else:
            qa_prefix = f"{question} {self.word_limit} words?\n {self.document_text} "
        if self.prompt_type == PromptType.MISTRAL:
            return self.prompt_mistral(qa_prefix)
        else:
            return self.prompt_adapt(qa_prefix)


def tag_question_augment(document_text: str, pos_tag: str,
                         document_type: str = "e-mail"):
    return ContextPromptingInput(document_type)(document_text, pos_tag)


def document_requests_augment(document_text: str, document_requests: List[str],
                              document_type: str = "e-mail"):
    return ContextPromptingInput(document_type)(document_text,
                                                document_requests)


def summarization_augment(document_text: str,
                          document_type: str = "e-mail", word_limit: int = 50):
    return SummarizePromptInput(word_limit=word_limit)(document_text,
                                                       document_type)
