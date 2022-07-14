import os
import argparse
import json
import ruamel.yaml as yaml

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

import utils
from trainer.dataset import create_dataset, create_loader
from trainer.dataset.utils import collect_result
from models.tokenization_bert import BertTokenizer
from models.model_vqa import XVLM


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

    for n, (image, question, question_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        if n < int(5000 / 160):
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


def main(args, config):
    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    samplers = [None]
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print("Creating vqa datasets")
    _, vqa_test_dataset = create_dataset('vqa', config, evaluate=False)
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

    vqa_result = evaluation(model, test_loader, tokenizer, device, config)

    result_rpath = collect_result(vqa_result, 'vqa_eval', local_wdir=args.result_dir,
                                  hdfs_wdir=None,
                                  write_to_hdfs=False, save_result=True)
    calculate_acc(result_rpath, vqa_test_dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='checkpoints/model_state_epoch_9.th')
    parser.add_argument('--config', default='./configs/VQA_480.yaml')
    parser.add_argument('--output_dir', default='output/vqa')
    args = parser.parse_args()
    args.result_dir = os.path.join(args.output_dir, 'result')
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    main(args, config)
