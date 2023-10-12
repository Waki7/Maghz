import random

from tqdm import tqdm
from transformers import PreTrainedTokenizer

from mgz.model_running.run_ops import embedding_controller, \
    tagging_embedding_controller
from mgz.models.nlp.base_transformer import BaseTransformer
from mgz.models.nlp.led import LEDForBinaryTagging
from mgz.typing import *
from mgz.version_control import ModelNode, lookup_model

# from transformers import BartConfig, LEDConfig

if torch.backends.mps.is_available():
    print('MPS available, will be running on apple GPU')
    torch.backends.mps.is_available()
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE_NUM = 0
    print('{} gpus available, will be using gpu {}'.format(
        torch.cuda.device_count(), DEVICE_NUM))
    DEVICE = torch.device('cuda:{}'.format(DEVICE_NUM))
    torch.cuda.set_device(DEVICE)
else:
    print('using cpu')
    DEVICE = torch.device('cpu')

TOKENIZERS: Dict[str, PreTrainedTokenizer] = {}
MODELS: Dict[str, BaseTransformer] = {}
MODEL_NAME = "/Users/ceyer/Documents/Projects/Maghz/index_dir/main/allenai/led-base-16384-multi_lexsum-source-long/train_step_18215_data_data_cls-AusCaseReportsToTagGrouped-_valacc_None/"


def load_models():
    print('loading model {}... '.format(MODEL_NAME))
    model_node: ModelNode = lookup_model(LEDForBinaryTagging, MODEL_NAME,
                                         MODEL_NAME)
    model: LEDForBinaryTagging = model_node.model
    tokenizer: PreTrainedTokenizer = model_node.tokenizer
    model.to(DEVICE)
    model.eval()
    MODELS[model_node.model_id] = model
    TOKENIZERS[model_node.model_id] = tokenizer
    print('loaded model {}... '.format(MODEL_NAME))


def encode(text: str) -> Tuple[
    NDArray['EmbedLen'], NDArray['SrcSeqLen*EmbedLen']]:
    with torch.no_grad():
        torch.manual_seed(5)
        logits: FloatTensorT['B,SrcSeqLen,EmbedLen'] = \
            embedding_controller(model=MODELS[MODEL_NAME], text=[text, ],
                                 tokenizer=TOKENIZERS[MODEL_NAME],
                                 get_last_embedding=False)
        embedding: FloatTensorT['SrcSeqLen,EmbedLen'] = logits[0]
        last_embedding: FloatTensorT['EmbedLen'] = logits[0][-1]
        return last_embedding.cpu().numpy(), embedding.flatten().cpu().numpy()


def get_tag_apply_embedding(doc_text: str, tag_text: str) -> NDArray[
    'EmbedLen']:
    with torch.no_grad():
        model: LEDForBinaryTagging = MODELS[MODEL_NAME]
        torch.manual_seed(5)
        logits: FloatTensorT['B,EmbedLen'] = \
            tagging_embedding_controller(model=model, text=[doc_text, ],
                                         tag_text=[tag_text, ],
                                         tokenizer=TOKENIZERS[MODEL_NAME], )
        embedding: FloatTensorT['EmbedLen'] = logits.flatten()
        return embedding.cpu().numpy()


def predict_tag(association_embedding: NDArray['EmbedLen'],
                positive_embedding: NDArray[
                    'EmbedLen'], negative_embedding: NDArray[
            'EmbedLen']) -> bool:
    association_embedding = FloatTensorT(association_embedding)
    positive_embedding = FloatTensorT(positive_embedding)
    negative_embedding = FloatTensorT(negative_embedding)

    distance_from_pos = torch.linalg.norm(
        association_embedding - positive_embedding, dim=-1, ord=2)
    distance_from_neg = torch.linalg.norm(
        association_embedding - negative_embedding, dim=-1, ord=2)

    probabilities: FloatTensorT['2'] = FloatTensorT(torch.softmax(
        -1 * torch.stack([distance_from_neg, distance_from_pos], dim=-1),
        dim=-1))
    # pos is in index 0, pos in index 1, so when we take the argmax, we get 0
    # (false) or 1 (true)
    return bool(probabilities.argmax(-1).item())


def importing_sample_data() -> List[Dict[str, Union[int, str, FloatTensorT]]]:
    from mgz.ds.sentence_datasets.aus_legal_case_reports import SampleType, \
        AusCaseReportsToTagGrouped
    from transformers import BartTokenizer
    import transformers as hug
    cfg: hug.LEDConfig = MODELS[MODEL_NAME].config
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    # tokenizer = TOKENIZERS[MODEL_NAME].config

    ds = AusCaseReportsToTagGrouped(tokenizer,
                                    max_src_len=cfg.max_encoder_position_embeddings,
                                    n_episodes=1000,
                                    n_shot=5).load_validation_data()
    ds_samples = ds.data

    return [
        {
            "id": 0,
            "source_text": ds_samples[1][SampleType.INPUT_TEXT],
            "tags": ds_samples[1][SampleType.CATCHPHRASES]
        },
        {
            "id": 1,
            "source_text": ds_samples[2][SampleType.INPUT_TEXT],
            "tags": ds_samples[2][SampleType.CATCHPHRASES]
        },
        {
            "id": 2,
            "source_text": ds_samples[3][SampleType.INPUT_TEXT],
            "tags": ds_samples[3][SampleType.CATCHPHRASES]
        },
        {
            "id": 3,
            "source_text": ds_samples[4][SampleType.INPUT_TEXT],
            "tags": ds_samples[4][SampleType.CATCHPHRASES]
        },
        {
            "id": 4,
            "source_text": ds_samples[5][SampleType.INPUT_TEXT],
            "tags": ds_samples[5][SampleType.CATCHPHRASES]
        },
        {
            "id": 5,
            "source_text": ds_samples[6][SampleType.INPUT_TEXT],
            "tags": ds_samples[6][SampleType.CATCHPHRASES]
        },
        {
            "id": 6,
            "source_text": ds_samples[7][SampleType.INPUT_TEXT],
            "tags": ds_samples[7][SampleType.CATCHPHRASES]
        },
        {
            "id": 7,
            "source_text": ds_samples[8][SampleType.INPUT_TEXT],
            "tags": ds_samples[8][SampleType.CATCHPHRASES]
        },
        {
            "id": 8,
            "source_text": ds_samples[9][SampleType.INPUT_TEXT],
            "tags": ds_samples[9][SampleType.CATCHPHRASES]
        },
    ]


def main():
    load_models()
    sample_tags = []
    [sample_tags.extend(sample['tags']) for sample in importing_sample_data()]
    correct = 0
    total = 0
    sample_data = importing_sample_data()
    sampling = 10
    for _ in tqdm(range(0, sampling)):
        # select random positive
        rand_pos = random.choice(sample_data)
        pos_id = rand_pos['id']
        random_tag = random.choice(rand_pos['tags'])

        # select random negative
        rand_neg = random.choice(sample_data)
        neg_id = rand_neg['id']
        while random_tag not in rand_neg['tags'] and rand_neg != rand_pos:
            rand_neg = random.choice(sample_data)

        random_tag = random.choice(sample_tags)
        embeddings = [get_tag_apply_embedding(sample['source_text'],
                                              random_tag) for sample in
                      sample_data]

        pos_center = embeddings[pos_id]
        neg_center = embeddings[neg_id]
        for i, sample in enumerate(sample_data):
            embedding = embeddings[i]
            if i not in (pos_id, neg_id):
                total += 1
                pred = predict_tag(embedding, pos_center, neg_center)
                correct += int(pred == (random_tag in sample['tags']))

    load_models()


if __name__ == '__main__':
    main()
