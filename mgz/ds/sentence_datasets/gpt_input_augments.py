from __future__ import annotations

from typing import List
from typing import TYPE_CHECKING

import transformers as hug

if TYPE_CHECKING:
    pass
GPT_TURN = f"<|end_of_turn|>GPT4 Correct Assistant:"


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
                 question_wording_component: Optional[str] = None, ):
        if prompt_type is None:
            self.prompt_type = self.infer_prompt_type(model)
        else:
            self.prompt_type = prompt_type
        self.system_context = system_context
        self.truncate_token_start = '[TRUNCABTABLE_START]'
        self.truncate_token_end = '[TRUNCABTABLE_END]'
        self.question_wording_component = question_wording_component


class BatchChatInput:

    def __init__(self,
                 system_context: str = "",
                 add_trunc=True,
                 **kwargs,
                 ):
        self.system_context = system_context
        self.truncate_token_start = '[TRUNCABTABLE_START]'
        self.truncate_token_end = '[TRUNCABTABLE_END]'
        self.add_trunc = add_trunc

    def get_tokenizer_input(self, tokenizer: hug.PreTrainedTokenizerBase):
        raise NotImplementedError(
            "This method should be implemented in the child class")


class DocumentTagChat(BatchChatInput):
    def __init__(self,
                 document_texts: List[str],
                 tags: List[List[str]],
                 document_type: str = "e-mail",
                 system_context: str = "",
                 add_trunc=True,
                 **kwargs):
        super().__init__(system_context=system_context, add_trunc=add_trunc)
        self.document_texts = document_texts
        self.tags = tags
        self.document_type = document_type


class DocumentRequestChat(BatchChatInput):
    def __init__(self, document_texts: List[str],
                 document_requests: List[List[str]],
                 document_type: str = "e-mail",
                 system_context: str = "",
                 add_trunc=True):
        super().__init__(system_context=system_context, add_trunc=add_trunc)
        self.document_texts = document_texts
        self.document_requests = document_requests
        self.document_type = document_type

    def get_tokenizer_input(self, tokenizer: hug.PreTrainedTokenizerBase) -> \
            List[str]:
        # return f"We are looking for {tag}. Is this {self.document_type} what we are looking for. Yes or no.\n"
        # return f"Is it apparent that the {self.document_type} is part of \"{tag}\"? Yes or no.\n"
        #     return f"Is the {self.document_type} part of \"{tag}\"? Answer yes or no"
        #     return f"Does the tag \"{tag}\", confidently apply to this {self.document_type}? Yes or no.\n"

        # return f"Is the {self.document_type} part of \"{tag}\"? Answer yes or no"
        tokenized_output = []
        for document_text in self.document_texts:
            if self.add_trunc:
                doc_context = f"{self.truncate_token_start}{document_text}{self.truncate_token_end}"
            else:
                doc_context = f" {document_text} "
            doc_request_questions = [
                f"Based on the contents of this {self.document_type}, is there discussion that categorizes it as \"{doc_request}\"?"
                for doc_request
                in self.document_requests]
            doc_request_question = ' '.join(doc_request_questions) + ' '
            if isinstance(tokenizer, hug.LlamaTokenizer) or isinstance(
                    tokenizer, hug.LlamaTokenizerFast):
                messages = [{
                    "role": "user",
                    "content": f"{self.system_context}\n{doc_context}\n{doc_request_question}"
                }]
            else:
                messages = [
                    {
                        "role": "user",
                        "content": f"{self.system_context}",

                    }, {
                        "role": "user",
                        "content": f"{doc_context}\n{doc_request_question}"
                    }]
            tokenized_output.append(tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            ))
        return tokenized_output


class SummarizePromptInput(BatchChatInput):
    def __init__(self, document_texts: List[str],
                 word_limit: int = 50,
                 document_type: str = "e-mail",
                 system_context: str = "", add_trunc=True):
        super().__init__(system_context=system_context, add_trunc=add_trunc)
        self.document_texts = document_texts
        self.word_limit = word_limit
        self.document_type = document_type

    def get_tokenizer_input(self, tokenizer: hug.PreTrainedTokenizerBase):
        if self.add_trunc:
            qa_prefix = f"Could you summarize this in at most {self.word_limit} words?:\n{self.truncate_token_start}{self.document_text}{self.truncate_token_end}"
        else:
            qa_prefix = f"Could you summarize this in at most {self.word_limit} words?:\n {self.document_text} "
        return tokenizer.apply_chat_template(
            [{"role": "user",
              "content": qa_prefix}],
            tokenize=False,
            add_generation_prompt=True
        )


class FreePromptInput(BatchChatInput):
    def __init__(self, raw_chats: List[str],
                 system_context: str = "", add_trunc=True,
                 **kwargs):
        super().__init__(system_context=system_context, add_trunc=add_trunc)
        self.raw_chats = raw_chats

    def get_tokenizer_input(self, tokenizer: hug.PreTrainedTokenizerBase):
        # return f"We are looking for {tag}. Is this {self.document_type} what we are looking for. Yes or no.\n"
        # return f"Is it apparent that the {self.document_type} is part of \"{tag}\"? Yes or no.\n"
        #     return f"Is the {self.document_type} part of \"{tag}\"? Answer yes or no"
        #     return f"Does the tag \"{tag}\", confidently apply to this {self.document_type}? Yes or no.\n"

        # return f"Is the {self.document_type} part of \"{tag}\"? Answer yes or no"
        tokenized_output = []
        for chat in self.raw_chats:
            if self.add_trunc:
                chat = f"{self.truncate_token_start}{chat}{self.truncate_token_end}"
            else:
                chat = f" {chat} "
            messages = [
                {
                    "role": "user",
                    "content": f"{chat}",
                }, ]
            tokenized_output.append(tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            ))
        return tokenized_output


def main():
    import transformers as hug
    tokenizer = hug.PreTrainedTokenizerFast.from_pretrained(
        # "mistralai/Mistral-7B-Instruct-v0.2",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        token="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")
    messages = [
        {"role": "system",
         "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": "Who are you?"},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    print(prompt)


if __name__ == '__main__':
    main()
