import os
import numpy as np
import json
import argparse
from init_args import ManagerArguments
import logging
from manager import Manager
from collections import Counter

class QATask:
    def __init__(self, file_dict, split="test"):
        self.file_dict = file_dict
        with open(self.file_dict[split]) as fin:
            self.data = [json.loads(line) for line in fin]
            self.task2id = {d["question"]: k for k, d in enumerate(self.data)}
        if "train" in self.file_dict:
            with open(self.file_dict["train"]) as fin:
                self.data_train = [json.loads(line) for line in fin]

def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for language models")

    parser.add_argument("--dataname", default='hotpotqa', help="")
    parser.add_argument("--train_file", default='', metavar="FILE", help="")
    parser.add_argument("--dev_file", default='', metavar="FILE", help="")
    parser.add_argument("--test_file", default='', metavar="FILE", help="")
    parser.add_argument("--eval_file", default='', metavar="FILE", help="")
    parser.add_argument("--split", default='train', help="")
    parser.add_argument("--eval", default=1, type=int, help="")
    args = parser.parse_args()

    manager_args = ManagerArguments()
    task_args = args
    return manager_args, task_args

import re, string
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
  
    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))

def f1_score(prediction, ground_truth):
    ZERO_METRIC = (0, 0, 0)
    prediction_tokens = prediction.split()
    ground_truth_tokens = ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def get_metrics(pred, truth):
    pred = normalize_answer(pred)
    gt = normalize_answer(truth)
    em = (pred == gt)
    f1, p, r = f1_score(pred, gt)
    return {'reward': em, 'em': em, 'f1': f1, 'recall': r, 'precision': p}


def eval(file_path):
    em_list, f1_list = [], []
    fin = open(file_path)
    for line in fin:
        g = json.loads(line)
        metric = get_metrics(pred=g["pred"], truth=g["answer"])

        em_list.append(metric["em"])
        f1_list.append([metric["precision"], metric["recall"], metric["f1"]])
    avg_em = np.mean(em_list)
    avg_f1 = np.mean(f1_list, 0)
    print(len(em_list), len(f1_list))
    print("avg em:\t\t", avg_em)
    print("avg f1:\t\t", avg_f1)

def run(manager_args, task_args):
    if task_args.eval:
        eval(task, file_path="./result.json")
    else:
        if manager_args.log and os.path.exists(manager_args.log):
            os.system("rm -r %s"%manager_args.log)
            os.makedirs(os.path.dirname(manager_args.log), exist_ok=True)

        logging.basicConfig(filename=manager_args.log, level=logging.INFO, format='''%(asctime)s %(pathname)s %(filename)s %(funcName)s %(lineno)s %(levelname)s - %(message)s''', filemode='a')

        file_dict = {"train": task_args.train_file, "dev": task_args.dev_file, "test": task_args.test_file}
        task = QATask(file_dict=file_dict, split=task_args.split)
        manager = Manager(args=manager_args)
        with open(f"../result/explore_{task_args.split}.json", "w") as fout:
            task_em, task_f1, task_recall = [], [], []
            for i in range(len(task.data)):
                result, trajs = manager.Solve_Executor(task=task.data[i]["question"])
                pred = result
                metric = get_metrics(pred=pred, truth=task.data[i]["answer"])

                task_em.append(metric["em"])
                task_f1.append(metric["f1"])
                task_recall.append(metric["recall"])
                avg_em = sum(task_em) / len(task_em)
                avg_f1 = sum(task_f1) / len(task_f1)
                avg_recall = sum(task_recall) / len(task_recall)

                print("question:", task.data[i]["question"])
                print("truth:", task.data[i]["answer"])
                print("pred:", result)
                print(i, 'len(task_em)', len(task_em), 'avg_em: %.4f\t\tavg_f1: %.4f\t\tavg_recall: %.4f'%(avg_em, avg_f1, avg_recall), '\n')
                print("="*10)
                fout.write(json.dumps({
                    "_id": task.data[i]['_id'] if "_id" in task.data[i] else task.data[i]["id"],
                    "question": task.data[i]['question'],
                    "answer": task.data[i]['answer'],
                    "pred": pred,
                    "traj": trajs,
                }) + "\n")
                print("WRITE ROW JSON DONE")

if __name__ == '__main__':
    manager_args, task_args = parse_args()
    run(manager_args, task_args)
