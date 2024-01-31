from __future__ import annotations

from typing import List, Optional

GPT_TURN = f"<|end_of_turn|>GPT4 Correct Assistant:"


class InputAugment:

    @staticmethod
    def prompt_adapt_mistral(main_body: str,
                             system_context: Optional[str] = "",
                             additional_chat: List[str] = []):
        """
        <s>[INST] <<SYS>>
        {{ system_prompt }}
        <</SYS>>

        {{ user_msg_1 }} [/INST] {{ model_answer_1 }} </s><s>[INST] {{ user_msg_2 }} [/INST]

        """
        if len(system_context) > 0:
            raise NotImplementedError("System context not implemented yet")
        prompt = f"GPT4 Correct User: " \
                 f"{main_body}\n" \
                 f"<|end_of_turn|>\n" \
                 f"GPT4 Correct Assistant: "
        for chat in additional_chat:
            prompt += f"<|end_of_turn|>" \
                      f"GPT4 Correct User: " \
                      f"{chat}\n" \
                      f"GPT4 Correct Assistant: "
        return prompt

    @staticmethod
    def prompt_adapt(main_body: str,
                     system_context: Optional[str] = "",
                     additional_chat: List[str] = []):
        """
        <s>[INST] <<SYS>>
        {{ system_prompt }}
        <</SYS>>

        {{ user_msg_1 }} [/INST] {{ model_answer_1 }} </s><s>[INST] {{ user_msg_2 }} [/INST]

        """

        prompt = f"<s>[INST] <<SYS>>{system_context}<</SYS>>\n\n{main_body} [/INST]"
        for chat in additional_chat:
            prompt += f" </s><s>[INST]" \
                      f"{chat} [/INST]"
        return prompt

    def __init__(self, document_type: str = "e-mail"):
        self.document_type = document_type

    def __call__(self, *args, **kwargs):
        pass


class ContextInputAugment(InputAugment):
    def __call__(self, document_text: str, document_requests: str | List[str]):
        if isinstance(document_requests, str):
            qa_prefix = f"\nGPT4 Correct User: Does the tag \"{document_requests}\" apply to the {self.document_type}?\n"
            qa_suffix = (GPT_TURN)
            return InputAugment.prompt_adapt(
                qa_prefix + document_text + '\n' + qa_suffix)
        else:
            full_prompt_text: str = ""
            full_prompt_text += f"GPT4 Correct User: Based on this {self.document_type}: \n {document_text} \n"
            for i, document_request in enumerate(document_requests):
                if i != 0:
                    full_prompt_text += f"GPT4 Correct User: "
                qa_next = f"Does the tag \"{document_request}\" apply to the {self.document_type}? Please answer in one word, and if you are not sure, say no.\n"
                qa_suffix = (GPT_TURN)
                if i != len(document_requests) - 1:
                    qa_suffix += "<|end_of_turn|>"
                full_prompt_text += qa_next + qa_suffix
            return full_prompt_text


def tag_question_augment(document_text: str, pos_tag: str,
                         document_type: str = "e-mail"):
    return ContextInputAugment(document_type)(document_text, pos_tag)


def document_requests_augment(document_text: str, document_requests: List[str],
                              document_type: str = "e-mail"):
    return ContextInputAugment(document_type)(document_text, document_requests)


class SummarizeAugment(InputAugment):
    def __init__(self, document_type: str = "e-mail", word_limit: int = 50):
        super().__init__(document_type)
        self.word_limit = word_limit

    def __call__(self, document_text: str,
                 document_type: str = "e-mail"):
        qa_prefix = f"GPT4 Correct User: Could you summarize this {document_type} in less than {self.word_limit} words? "
        qa_suffix = (GPT_TURN)
        # qa_suffix = (f"<|end_of_turn|>GPT4 Correct Assistant: "
        #              f"Could you summarize this {document_type} in less than {word_limit} words? ")  # last line is prompting because it seems to always start with this line (mistral openchat 3.5 does)
        return qa_prefix + document_text + qa_suffix


def summarization_augment(document_text: str,
                          document_type: str = "e-mail", word_limit: int = 50):
    return SummarizeAugment(word_limit=word_limit)(document_text, document_type)
