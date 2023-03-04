# from mgz.models.nlp.bart_interface import BARTHubInterface
from spacy.language import Language

def main2():
    from transformers import BartConfig
    from transformers import BartForConditionalGeneration, BartTokenizer

    use_mgz = True
    # Initializing a BART facebook/bart-large style configuration
    configuration = BartConfig()
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    if use_mgz:
        from mgz.models.nlp.bart import BartModel
        input_ids = tokenizer("Hello world </s>",
                              return_tensors="pt").input_ids  # Batch size 1
        model = BartModel(configuration)
        model.forward(input_ids)
        model.generate(input_ids)

    else:
        from transformers import BartModel
        # Initializing a model (with random weights) from the facebook/bart-large style configuration
        model = BartModel(configuration)
    # print(model)
    # Accessing the model configuration
    configuration = model.config


    # import torchtext
    # from torchtext.data import get_tokenizer
    # tokenizer: Language = get_tokenizer("basic_english")

    print(tokenizer.__call__(" Hello world"))
    print(tokenizer.__call__("Hello world"))

    print(tokenizer.tokenize("Hello world </s>"))
    print(tokenizer.__call__("Hello world </s>"))
    print(tokenizer.__call__("Hello world asdf"))
    print(tokenizer.__call__("Hello world <pad>>"))


    input_ids = tokenizer("Hello world </s>", return_tensors="pt").input_ids  # Batch size 1
    # print(model.forward(input_ids))
# def main():
#     from mgz.models.nlp.bart import BARTModel
#     bart_large = "C:/Users/ceyer/.cache/torch/pytorch_fairseq/40858f8de84f479771b2807266d806749e9ad0f8cb547921c35a76ae9c3ed0f6.099ef973524a5edb31b1211569b67bcc2863bc6d00781b79bac752acf8e48991/model.pt"
#     import torch
#     bart = BARTModel().load_state_dict(torch.load(bart_large))
#     print(bart)
#     # bart: BARTHubInterface = torch.hub.load('pytorch/fairseq', 'bart.large')
#     print(type(bart))
#     bart.eval()  # disable dropout (or leave in train mode to finetune)
#     tokens = bart.encode('Hello world!')
#     assert tokens.tolist() == [0, 31414, 232, 328, 2]
#     bart.decode(tokens)  # 'Hello world!'
#     print(bart.decode(tokens))


if __name__ == '__main__':
    main2()