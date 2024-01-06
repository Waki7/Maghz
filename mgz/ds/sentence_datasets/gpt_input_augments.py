def tag_question_augment(document_text: str, pos_tag: str,
                         document_type: str = "e-mail"):
    qa_prefix = f"GPT4 Correct User: Does the tag \"{pos_tag}\" apply to the {document_type}, answer in one word? "
    qa_suffix = (f"<|end_of_turn|>GPT4 Correct Assistant: "
                 f"Does the tag \"{pos_tag}\" apply to this {document_type}, answer in one word? ")  # last line is prompting because it seems to always start with this line (mistral openchat 3.5 does)
    return qa_prefix + document_text + qa_suffix


def one_word_answer_question_augment(document_text: str, content: str):
    qa_prefix = f"GPT4 Correct User: {content}, answer in one word? \n"
    qa_suffix = (f"<|end_of_turn|>GPT4 Correct Assistant: "
                 f"{content}, answer in one word? ")  # last line is prompting because it seems to always start with this line (mistral openchat 3.5 does)
    return qa_prefix + document_text + qa_suffix


def summarization_augment(document_text: str,
                          document_type: str = "e-mail", n_words: int = 50):
    qa_prefix = f"GPT4 Correct User: Could you summarize this {document_type} in less than {n_words} words? "
    qa_suffix = (f"<|end_of_turn|>GPT4 Correct Assistant: "
                 f"Could you summarize this {document_type} in less than {n_words} words? ")  # last line is prompting because it seems to always start with this line (mistral openchat 3.5 does)
    return qa_prefix + document_text + qa_suffix
