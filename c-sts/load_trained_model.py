from transformers import RobertaTokenizer

from utils.sts.modeling_encoders import TriEncoderForClassification


def get_model_and_tokenizer(device, path, config):
    tokenizer = RobertaTokenizer.from_pretrained(path)
    model = TriEncoderForClassification.from_pretrained(path, config=config).to(device)
    return model, tokenizer
