"""Handler for answering question from a specified image."""
import argparse
import ruamel.yaml as yaml
import json
import math
import torch
import torch.nn as nn
import pandas as pd

from PIL import Image
from torchvision import transforms

from models.model_vqa import XVLM
from models.tokenization_bert import BertTokenizer

from utils import pre_question


def parse_args():
    """Parse default arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='checkpoints/model_state_epoch_9.th')
    parser.add_argument('--config', default='configs/VQA_480.yaml')
    parser.add_argument('--image_path', default='images/random_2.jpg')
    parser.add_argument('--question', default='What color is the bench?')
    parser.add_argument('--n_repeats', default=16)
    args = parser.parse_args()
    return args


def analyze_uncertainty(topk_ids, topk_logits, topk_probs, config):
    """Analyze uncertainty from the topk logits."""
    args = parse_args()
    answer_list = json.load(open(config['answer_list'], 'r'))
    flattened_ids = topk_ids.view(args.n_repeats * config['k_test'])
    flattend_answers = [answer_list[id] for id in flattened_ids.cpu().numpy().astype(int)]
    flattened_probs = topk_probs.view(args.n_repeats * config['k_test'])
    flattened_logits = topk_logits.view(args.n_repeats * config['k_test'])

    all_answer_df = pd.DataFrame({"Predictions": flattend_answers,
                                  "Logits": flattened_logits.cpu().numpy(),
                                  "Probabilities": flattened_probs.cpu().numpy()})

    # all_answer

    mean_df = all_answer_df.groupby("Predictions").mean()
    deviation_df = all_answer_df.groupby("Predictions").std(ddof=0)
    mean_df["Deviations"] = deviation_df["Probabilities"]
    answer_df = mean_df.sort_values(by=['Probabilities'], ascending=False)
    top_5_answers = answer_df.head(5)
    top_5_preds = top_5_answers.index

    return top_5_answers, all_answer_df[all_answer_df.Predictions.isin(top_5_preds)]


def answer_question(model, image, question, tokenizer, device, config):
    """Generate caption for the specified image."""
    model.eval()
    answer_list = json.load(open(config['answer_list'], 'r'))
    answer_input = tokenizer(answer_list, padding='longest', return_tensors='pt').to(device)

    image = image.to(device, non_blocking=True)
    question_input = tokenizer(question, padding='longest', return_tensors="pt").to(device)

    with torch.no_grad():
        topk_ids, topk_probs, topk_logits = model(image=image, question=question_input, answer=answer_input, train=False, k=config['k_test'])

    # result = []
    # for topk_id, topk_prob in zip(topk_ids, topk_probs):
    #     prob, pred = topk_prob.max(dim=0)
    #     prob = math.floor(prob.item() * 1e4) / 100  # round down to two decimals
    #     result.append({"answer": answer_list[topk_id[pred]], "probability": prob})

    uncertainty, all_answers = analyze_uncertainty(topk_ids, topk_logits, topk_probs, config)
    return uncertainty, all_answers


def prepare_image(image, config) -> torch.Tensor:
    """Prepare image for answering."""
    image = image.convert('RGB')
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                     (0.26862954, 0.26130258, 0.27577711))
    transform = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.Resampling.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])
    image = transform(image)
    image = image.unsqueeze(dim=0)
    image = image.view(1, 3, config['image_res'], config['image_res'])
    return image


def prepare_question(question_txt):
    """Prepare question for the text encoder."""
    max_ques_words = 30
    question = pre_question(question_txt, max_ques_words)
    return question


def load_model(model_path, config) -> torch.nn.Module:
    """Load model from specified checkpoint."""
    model = XVLM(config=config)
    model.load_pretrained(model_path, config, is_eval=True)
    model.eval()
    return model


def answering_handler():
    """Handle image captioning."""
    args = parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image = Image.open(args.image_path).convert('RGB')
    image = prepare_image(image, config)
    image = image.to(device)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    config['pad_token_id'] = tokenizer.pad_token_id
    config['eos'] = '[SEP]'
    model = load_model(args.model_path, config)
    if torch.cuda.is_available():
        model = nn.DataParallel(model)
    model = model.to(device)

    question = args.question
    question = prepare_question(question)

    answer = answer_question(model, image, question, tokenizer, device, config)
    print(answer)


def answer_api(image, question):
    """Caption image from app."""
    args = parse_args()
    n_repeats = args.n_repeats
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device("cpu")

    image = Image.open(args.image_path).convert('RGB')
    image = prepare_image(image, config)
    image = image.repeat(n_repeats, 1, 1, 1)
    image = image + .2 * torch.normal(0, 1, size=image.shape)
    image = image.to(device)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    config['pad_token_id'] = tokenizer.pad_token_id
    config['eos'] = '[SEP]'
    model = load_model(args.model_path, config)
    if device == torch.device('cuda'):
        model = nn.DataParallel(model)
    model = model.to(device)

    question = prepare_question(question)
    question = [question] * n_repeats
    answer = answer_question(model, image, question, tokenizer, device, config)

    return answer


if __name__ == "__main__":
    answering_handler()
