import os
import argparse
import json
from numpy import dtype, uint8
import ruamel.yaml as yaml
from PIL import Image

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms as T
from trainer.dataset.utils import pre_question

import utils
from trainer.dataset import create_dataset, create_loader
from trainer.dataset.utils import collect_result
from models.tokenization_bert import BertTokenizer
from models.model_vqa import XVLM

from transformers import MarianMTModel, MarianTokenizer


def translate(texts, model, tokenizer, language="fr"):
    # Prepare the text data into appropriate format for the model
    def template(text): return f"{text}" if language == "en" else f">>{language}<< {text}"
    src_texts = [template(text) for text in texts]

    # Tokenize the texts
    # encoded = tokenizer.prepare_seq2seq_batch(src_texts)
    encoded = tokenizer.prepare_seq2seq_batch(src_texts, return_tensors="pt")

    # Generate translation using model
    translated = model.generate(**encoded)

    # Convert the generated tokens indices back into text
    translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)

    return translated_texts


def back_translate(texts, target_model, target_tokenizer, en_model, en_tokenizer, source_lang="en", target_lang="fr"):
    # Translate from source to target language
    fr_texts = translate(texts, target_model, target_tokenizer,
                         language=target_lang)

    # Translate from target language back to source language
    back_translated_texts = translate(fr_texts, en_model, en_tokenizer,
                                      language=source_lang)

    return back_translated_texts


@torch.no_grad()
def robust_evaluation(model, data_loader, tokenizer, device, config):
    # test
    model.eval()
    augmenter = T.AugMix()
    # augmenter = T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
    robust_transform = T.Compose([
        T.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
        augmenter,
        T.ToTensor(),
    ])

    target_model_name = 'Helsinki-NLP/opus-mt-en-ROMANCE'

    target_tokenizer = MarianTokenizer.from_pretrained(target_model_name)
    target_model = MarianMTModel.from_pretrained(target_model_name)

    en_model_name = 'Helsinki-NLP/opus-mt-ROMANCE-en'
    en_tokenizer = MarianTokenizer.from_pretrained(en_model_name)
    en_model = MarianMTModel.from_pretrained(en_model_name)

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate VQA test result:'
    print_freq = 50

    outputs = []

    answer_list = [answer + config['eos'] for answer in data_loader.dataset.answer_list]
    answer_input = tokenizer(answer_list, padding='longest', return_tensors='pt').to(device)

    for n, (image_paths, images, questions, question_ids) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        if n < int(5000 / 800):
            for image_path, image_tensor, question, ques_id in zip(image_paths, images, questions, question_ids):
                image = Image.open(image_path).convert('RGB')
                aug_images = [robust_transform(image) for _ in range(4)]
                # aug_images = [image_tensor for _ in range(4)]  # for testing without image augmentation
                aug_questions = []
                aug_questions.append(question)
                aug_question_1 = back_translate([question], target_model, target_tokenizer, en_model, en_tokenizer, target_lang="es")
                aug_questions.append(pre_question(aug_question_1[0], 50))
                aug_question_2 = back_translate([question], target_model, target_tokenizer, en_model, en_tokenizer, target_lang="fr")
                aug_questions.append(pre_question(aug_question_2[0], 50))
                aug_question_3 = back_translate([question], target_model, target_tokenizer, en_model, en_tokenizer, target_lang="it")
                aug_questions.append(pre_question(aug_question_3[0], 50))
                # utils.plot(aug_images, image_path, aug_questions)
                aug_images = torch.stack(aug_images, dim=0).to(device, non_blocking=True)
                # aug_questions = [question for _ in range(4)]  # for testing without text augmentation
                aug_question_input = tokenizer(aug_questions, padding='longest', return_tensors="pt").to(device)
                topk_ids, topk_probs = model(aug_images, aug_question_input, answer_input, train=False, k=config['k_test'])

                # average the probabilities grouped by the index
                c = topk_ids.view(-1).long()
                x = topk_probs.view(-1).unsqueeze(1)
                # allocate space for output
                result = torch.zeros((c.max() + 1, x.shape[1]), dtype=x.dtype).to(device)
                # use index_add_ to sum up rows according to tokens
                result.index_add_(0, c, x)
                result = result[result.sum(dim=1) != 0]
                # use "unique" to count how many of each token
                unique_id, counts = torch.unique(c, return_counts=True)
                # divide the sum by the counts to get the average probability for each token
                result /= counts[:, None]
                ques_id = int(ques_id.item())
                prob = result.max().item()
                top_id = unique_id[torch.argmax(result)].item()

                outputs.append({"question_id": ques_id, "answer": data_loader.dataset.answer_list[top_id], "probability": prob})

        else:
            break

    return outputs


@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate VQA test result:'
    print_freq = 50

    result = []

    answer_list = [answer + config['eos'] for answer in data_loader.dataset.answer_list]
    answer_input = tokenizer(answer_list, padding='longest', return_tensors='pt').to(device)

    for n, (_, image, question, question_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        if n < int(5000 / 800):
            image = image.to(device, non_blocking=True)
            question_input = tokenizer(question, padding='longest', return_tensors="pt").to(device)

            topk_ids, topk_probs = model(image, question_input, answer_input, train=False, k=config['k_test'])

            for ques_id, topk_id, topk_prob in zip(question_id, topk_ids, topk_probs):
                ques_id = int(ques_id.item())
                prob, pred = topk_prob.max(dim=0)
                prob = prob.item()
                result.append({"question_id": ques_id, "answer": data_loader.dataset.answer_list[topk_id[pred]], "probability": prob})

        else:
            break

    return result


def calculate_acc(result_rpath, test_dataset):
    print("Calculating accuracy")
    gt = {}
    for ann in test_dataset.ann:
        if 'answer' in ann.keys():
            gt[ann['question_id']] = ann['answer']
        else:
            return

    n = 0
    n_correct = 0
    with open(result_rpath, 'r') as f:
        for sample in json.load(f):
            n += 1
            index = sample['question_id']
            if sample['answer'].strip() in gt[index]:
                n_correct += 1

    print(f"n_questions: {n}, n_correct: {n_correct}", flush=True)
    if n > 0:
        print(f"acc: {n_correct / n}", flush=True)


def calculate_mean_unc(result_rpath, test_dataset, threshold=0.5):
    print("Calculating accuracy")
    gt = {}
    for ann in test_dataset.ann:
        if 'answer' in ann.keys():
            gt[ann['question_id']] = ann['answer']
        else:
            return

    n = 0
    n_unknowns = 0
    total_conf = 0.
    with open(result_rpath, 'r') as f:
        for sample in json.load(f):
            n += 1
            index = sample['question_id']
            total_conf += sample['probability']
            answer = sample['answer'].strip()
            if answer == 'unknown' or sample['probability'] < threshold:
                n_unknowns += 1

    # print(f"n_questions: {n}, n_unknowns: {n_unknowns}", flush=True)
    if n > 0:
        print(f"Threshold: {threshold}, unknwon %: {n_unknowns / n}, mean_conf: {total_conf / n}", flush=True)


def main(args, config):
    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    samplers = [None]
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print("Creating vqa datasets")
    _, vqa_test_dataset = create_dataset('vqa', config, evaluate=True)
    test_loader = create_loader([vqa_test_dataset], samplers,
                                batch_size=[config['batch_size_test']],
                                num_workers=[4], is_trains=[False],
                                collate_fns=[None])[0]

    config['pad_token_id'] = vqa_test_dataset.pad_token_id
    config['eos'] = vqa_test_dataset.eos_token
    model = XVLM(config=config)
    model.load_pretrained(args.model_path, config, is_eval=True)
    # model = nn.DataParallel(model)
    model = model.to(device)
    print("### Total Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    vqa_result = robust_evaluation(model, test_loader, tokenizer, device, config)
    # vqa_result = evaluation(model, test_loader, tokenizer, device, config)

    result_rpath = collect_result(vqa_result, 'vqa_eval', local_wdir=args.result_dir,
                                  hdfs_wdir=None,
                                  write_to_hdfs=False, save_result=True)

    calculate_acc(result_rpath, vqa_test_dataset)
    calculate_mean_unc(result_rpath, vqa_test_dataset, threshold=0.5)
    calculate_mean_unc(result_rpath, vqa_test_dataset, threshold=0.8)
    calculate_mean_unc(result_rpath, vqa_test_dataset, threshold=0.9)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='checkpoints/model_state_epoch_9.th')
    parser.add_argument('--config', default='./configs/VQA_480.yaml')
    parser.add_argument('--output_dir', default='output/vqa')
    args = parser.parse_args()
    args.result_dir = os.path.join(args.output_dir, 'result')
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    main(args, config)
