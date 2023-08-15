# Load model directly
def main():


    USE_LED = True
    if USE_LED:
        from transformers import LEDForConditionalGeneration, LEDTokenizerFast
        tokenizer = LEDTokenizerFast.from_pretrained(
            "allenai/led-base-16384-multi_lexsum-source-long")
        model = LEDForConditionalGeneration.from_pretrained(
            "allenai/led-base-16384-multi_lexsum-source-long")
        print(model)
        print(type(model))
        print(type(tokenizer))

if __name__ == '__main__':
    main()