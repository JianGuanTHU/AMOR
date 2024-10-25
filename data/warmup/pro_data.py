from os import replace
import requests
import copy
import numpy as np
np.random.seed(42)
import traceback
import re
import json
from prompts_ours import get_decompose_prompt, get_relevance_prompt, get_solve_prompt, get_finish_prompt


def retrieve_helper(data):
    resp = requests.post("http://0.0.0.0:8095/retriever_wikipedia", json=data)
    assert resp.status_code == 200, f"request failed: {resp.status_code}, {resp.text}"
    result = resp.json()["response"]
    return result

def pro_nq():
    def retrieve():
        with open("./nq/nq-train.jsonl") as fin:
            alld = []
            for line in fin:
                alld.append(json.loads(line))
        allquestions = [d["input"] for d in alld][:10000]
        with open("./nq/nq-train-searchdoc.json", "w") as fout:
            data = {
                "data": [{
                    "query": q,
                } for q in allquestions],
                "n_doc": 20,
                "func": "SearchDoc",
            }
            result = retrieve_helper(data)
            for d, r in zip(alld, result):
                d["ctxs"] = r["ctxs"]
                fout.write(json.dumps(d) + "\n")

    def retrievepsg():
        with open("./nq/nq-train-searchdoc.json") as fin:
            with open("./nq/nq-train-searchdoc-searchpsg.json", "w") as fout:
                for k, line in enumerate(fin):
                    d = json.loads(line)
                    true_titles = {}
                    answers = []
                    for opt in d["output"]:
                        if "answer" in opt:
                            answers.append(opt["answer"])
                        if "provenance" in opt:
                            for elist in opt["provenance"]:
                                true_titles[elist["title"]] = 1
                    for c in d["ctxs"]:
                        data = {
                            "data": [{
                                "query": d["input"],
                                "title": c["title"],
                            }],
                            "n_doc": 20,
                            "func": "SearchPsg",
                        }
                        result = retrieve_helper(data)
                        c["psg_ctxs"] = result[0]["ctxs"]
                    fout.write(json.dumps(d) + "\n")

    def make_train_data_ours():
        with open("./nq/nq-train-searchdoc-searchpsg.json") as fin:
            fout = open("./nq/nq-train-searchdoc-searchpsg-ours.json", "w")
            for idx1, line in enumerate(fin):
                d = json.loads(line)
                answers = {}
                true_titles = {}
                for opt in d["output"]:
                    if "answer" in opt:
                        answers[opt["answer"]] = 1
                    if "provenance" in opt:
                        for elist in opt["provenance"]:
                            true_titles[elist["title"]] = 1

                query = d["input"].strip()
                fout.write(json.dumps({
                        "id": d["id"],
                        "ipt_type": "task_decompose",
                        "opt_type": "next",
                        "ipt": {
                            "task": query,
                            "query": None,
                            "history": [],
                            "obs": None,
                            },
                        "opt": {"query": query.strip()}
                    })+ "\n")
                
                doc_ctxs = d["ctxs"]
                allctxs = [{key: ctx[key] for key in ["id", "title", "text"]} for ctx in doc_ctxs if ctx["title"] not in true_titles]
                if len(allctxs):
                    fout.write(json.dumps({
                            "id": d["id"],
                            "ipt_type": "relevance_judgment",
                            "opt_type": "irrelevant",
                            "ipt": {
                                "task": query,
                                "query": query,
                                "history": [],
                                "obs": np.random.choice(allctxs),
                                },
                            "opt": None,
                            "src": "contriever",
                        })+ "\n")
                allctxs = [{key: ctx[key] for key in ["id", "title", "text"]} for ctx in doc_ctxs if ctx["title"] in true_titles]
                if len(allctxs):
                    fout.write(json.dumps({
                            "id": d["id"],
                            "ipt_type": "relevance_judgment",
                            "opt_type": "relevant",
                            "ipt": {
                                "task": query,
                                "query": query,
                                "history": [],
                                "obs": np.random.choice(allctxs),
                                },
                            "opt": None,
                            "src": "contriever",
                        })+ "\n")

                for doc_ctx in doc_ctxs:
                    if doc_ctx["title"] not in true_titles:
                        continue
                    pos_ctxs, pos_ans_list, neg_ctxs = [], [], []
                    psg_ctxs = doc_ctx["psg_ctxs"]
                    for ctx in psg_ctxs:
                        flag = False
                        for ans in answers:
                            if ans in ctx["text"]:
                                pos_ctxs.append(ctx)
                                pos_ans_list.append(ans)
                                flag = True
                                break
                        if not flag:
                            neg_ctxs.append(ctx)
                    if len(pos_ctxs) == 1:
                        record_id = pos_ctxs[0]["id"]
                        mix_ctxs = np.random.permutation([pos_ctxs[0]] + neg_ctxs[:2]).tolist()
                        pos = None
                        for pos_, ctx in enumerate(mix_ctxs):
                            if ctx["id"] == record_id:
                                pos = pos_ + 1
                                break
                        fout.write(json.dumps({
                            "id": d["id"],
                            "ipt_type": "answer_extraction",
                            "opt_type": "answerable",
                            "ipt": {
                                    "task": query,
                                    "query": query,
                                    "history": [],
                                    "obs": mix_ctxs,
                                },
                            "opt": {"ID": pos, "answer": pos_ans_list[0]},
                            "src": "contriever",
                        })+ "\n")

                        fout.write(json.dumps({
                            "id": d["id"],
                            "ipt_type": "finish",
                            "opt_type": "finish",
                            "ipt": {
                                "task": query,
                                "history": [{"Q": query, "A": pos_ans_list[0]}],
                                "obs": pos_ctxs[:1],
                            },
                            "opt": {"result": pos_ans_list[0]},
                        }) + "\n")
                    if len(neg_ctxs):
                        neg_ctxs = np.random.permutation(neg_ctxs).tolist()[:3]
                        fout.write(json.dumps({
                            "id": d["id"],
                            "ipt_type": "answer_extraction",
                            "opt_type": "unanswerable",
                            "ipt": {
                                    "task": query,
                                    "query": query,
                                    "history": [],
                                    "obs": neg_ctxs,
                                },
                            "opt": None,
                            "src": "contriever",
                        })+ "\n")

                    fout.write(json.dumps({
                            "id": d["id"],
                            "ipt_type": "task_decompose",
                            "opt_type": "finish",
                            "ipt": {
                                "task": query,
                                "query": query,
                                "history": [{"Q": query, "A": np.random.choice(list(answers.keys()))}],
                                "obs": None,
                                },
                            "opt": None,
                        })+ "\n")

    retrieve()
    retrievepsg()
    make_train_data_ours()


def pro_boolq():
    def retrieve():
        with open("./boolq/train.jsonl") as fin:
            alld = []
            for line in fin:
                alld.append(json.loads(line))
        allquestions = [d["question"] for d in alld]
        with open("./boolq/train-searchdoc.json", "w") as fout:
            data = {
                "data": [{
                    "query": q,
                } for q in allquestions],
                "n_doc": 20,
                "func": "SearchDoc",
            }
            result = retrieve_helper(data)
            for d, r in zip(alld, result):
                d["ctxs"] = r["ctxs"]
                fout.write(json.dumps(d) + "\n")

    def retrievepsg():
        with open("./boolq/train-searchdoc.json") as fin:
            with open("./boolq/train-searchdoc-searchpsg.json", "w") as fout:
                for k, line in enumerate(fin):
                    d = json.loads(line)
                    for c in d["ctxs"]:
                        data = {
                            "data": [{
                                "query": d["question"],
                                "title": c["title"],
                            }],
                            "n_doc": 20,
                            "func": "SearchPsg",
                        }
                        result = retrieve_helper(data)
                        c["psg_ctxs"] = result[0]["ctxs"]
                    fout.write(json.dumps(d) + "\n")

    def match_golden():
        for name in ["train", "dev"]:
            with open(f"./boolq/{name}.jsonl") as fin:
                with open(f"./boolq/{name}_goldenpsg.json", "w") as fout:
                    for line in fin:
                        d = json.loads(line)
                        title = d["title"]
                        data = {
                            "data": [{
                                "query": d["passage"],
                                "title": title,
                            }],
                            "n_doc": 1,
                            "func": "SearchPsg",
                        }
                        try:
                            result = retrieve_helper()
                            for ctx in result[0]["ctxs"]:
                                d["golden_psg"] = ctx
                                break
                        except:
                            d["golden_psg"] = None
                        fout.write(json.dumps(d)+"\n")

    def make_train_data_ours():
        qid2goldenctx = {}
        with open(f"./boolq/train_goldenpsg.json") as fin:
            for k, line in enumerate(fin):
                d = json.loads(line)
                if d["golden_psg"] is None:
                    qid2goldenctx[k] = None
                else:
                    qid2goldenctx[k] = d["golden_psg"]

        with open("./boolq/train-searchdoc-searchpsg.json") as fin:
            fout = open("./boolq/train-searchdoc-searchpsg-ours.json", "w")
            for k, line in enumerate(fin):
                if qid2goldenctx[k] is None:
                    continue
                d = json.loads(line)
                answer = "yes" if d["answer"] else "no"
                true_titles = [d["title"]]

                query = d["question"].strip()
                fout.write(json.dumps({
                        "id": k,
                        "ipt_type": "task_decompose",
                        "opt_type": "next",
                        "ipt": {
                            "task": query,
                            "query": None,
                            "history": [],
                            "obs": None,
                            },
                        "opt": {"query": query.strip()}
                    })+ "\n")
                
                doc_ctxs = d["ctxs"]
                allctxs = [{key: ctx[key] for key in ["id", "title", "text"]} for ctx in doc_ctxs if ctx["title"] not in true_titles]
                if len(allctxs):
                    fout.write(json.dumps({
                            "id": k,
                            "ipt_type": "relevance_judgment",
                            "opt_type": "irrelevant",
                            "ipt": {
                                "task": query,
                                "query": query,
                                "history": [],
                                "obs": np.random.choice(allctxs),
                                },
                            "opt": None,
                            "src": "contriever",
                        })+ "\n")
                allctxs = [{key: ctx[key] for key in ["id", "title", "text"]} for ctx in doc_ctxs if ctx["title"] in true_titles]
                if len(allctxs):
                    fout.write(json.dumps({
                            "id": k,
                            "ipt_type": "relevance_judgment",
                            "opt_type": "relevant",
                            "ipt": {
                                "task": query,
                                "query": query,
                                "history": [],
                                "obs": np.random.choice(allctxs),
                                },
                            "opt": None,
                            "src": "contriever",
                        })+ "\n")

                for doc_ctx in doc_ctxs:
                    if doc_ctx["title"] not in true_titles:
                        continue
                    pos_ctxs, neg_ctxs = [], []
                    psg_ctxs = doc_ctx["psg_ctxs"]
                    for ctx in psg_ctxs:
                        if ctx["id"] == qid2goldenctx[k]["id"]:
                            pos_ctxs.append(ctx)
                        else:
                            neg_ctxs.append(ctx)
                    if len(pos_ctxs) == 1:
                        record_id = pos_ctxs[0]["id"]
                        mix_ctxs = np.random.permutation([pos_ctxs[0]] + neg_ctxs[:2]).tolist()
                        pos = None
                        for pos_, ctx in enumerate(mix_ctxs):
                            if ctx["id"] == record_id:
                                pos = pos_ + 1
                                break
                        fout.write(json.dumps({
                            "id": k,
                            "ipt_type": "answer_extraction",
                            "opt_type": "answerable",
                            "ipt": {
                                    "task": query,
                                    "query": query,
                                    "history": [],
                                    "obs": mix_ctxs,
                                },
                            "opt": {"ID": pos, "answer": answer},
                            "src": "contriever",
                        })+ "\n")

                        fout.write(json.dumps({
                            "id": k,
                            "ipt_type": "finish",
                            "opt_type": "finish",
                            "ipt": {
                                "task": query,
                                "history": [{"Q": query, "A": answer}],
                                "obs": pos_ctxs[:1],
                            },
                            "opt": {"result": answer},
                        }) + "\n")
                    if len(neg_ctxs):
                        neg_ctxs = np.random.permutation(neg_ctxs).tolist()[:3]
                        fout.write(json.dumps({
                            "id": k,
                            "ipt_type": "answer_extraction",
                            "opt_type": "unanswerable",
                            "ipt": {
                                    "task": query,
                                    "query": query,
                                    "history": [],
                                    "obs": neg_ctxs,
                                },
                            "opt": None,
                            "src": "contriever",
                        })+ "\n")

                    fout.write(json.dumps({
                            "id": k,
                            "ipt_type": "task_decompose",
                            "opt_type": "finish",
                            "ipt": {
                                "task": query,
                                "query": query,
                                "history": [{"Q": query, "A": answer}],
                                "obs": None,
                                },
                            "opt": None,
                        })+ "\n")


    retrieve()
    retrievepsg()
    match_golden()
    make_train_data_ours()


def pro_musique():
    def merge_question():
        with open("./musique/musique_train.jsonl") as fin:
            alld = []
            for line in fin:
                d = json.loads(line)
                qs = d["question_decomposition"]
                for idx2, q in enumerate(qs):
                    tgtq = q["question"]
                    for i in range(1,6):
                        if f"#{i}" in tgtq:
                            tgtq = tgtq.replace(f"#{i}", qs[i-1]["answer"])
                    q["merge_question"] = tgtq
                alld.append(d)

        for i in range(9):
            with open(f"./musique/train_question_{i}.json") as fin:
                for line in fin:
                    d = json.loads(line)
                    idx1, idx2 = d["idx1"], d["idx2"]
                    alld[idx1]["question_decomposition"][idx2]["_question"] = d["output"]

        with open("./musique/train_question_merge.json", "w") as fout:
            for d in alld:
                fout.write(json.dumps(d) + "\n")

    def retrieve():        
        with open("./musique/train_question_merge.json") as fin:
            allquestions = []
            for idx1, line in enumerate(fin):
                d = json.loads(line)
                qs = d["question_decomposition"]
                for idx2, q in enumerate(qs):
                    if "_question" in q:
                        allquestions.append({"idx1": idx1, "idx2": idx2, "type": "_question", "query": q["_question"]})
                    else:
                        allquestions.append({"idx1": idx1, "idx2": idx2, "type": "merge_question", "query": q["merge_question"]})
        with open("./musique/train_question_merge_searchdoc.json", "w") as fout:
            data = {
                "data": allquestions,
                "n_doc": 20,
                "func": "SearchDoc",
            }
            result = retrieve_helper(data)
            for q, r in zip(allquestions, result):
                q["ctxs"] = r["ctxs"]
                fout.write(json.dumps(q) + "\n")

    def retrievepsg():
        with open("./musique/train_question_merge_searchdoc.json") as fin:
            with open("./musique/train_question_merge_searchdoc_searchpsg.json", "w") as fout:
                for k, line in enumerate(fin):
                    d = json.loads(line)
                    for c in d["ctxs"]:
                        data = {
                            "data": [{
                                "query": d["query"],
                                "title": c["title"],
                            }],
                            "n_doc": 20,
                            "func": "SearchPsg",
                        }
                        result = retrieve_helper(data)
                        c["psg_ctxs"] = result[0]["ctxs"]
                    fout.write(json.dumps(d) + "\n")

    def make_train_data_ours():
        ctxs = {}
        with open("./musique/train_question_merge_searchdoc_searchpsg.json") as fin:
            for k, line in enumerate(fin):
                if k % 1000 == 0:
                    print(k)
                d = json.loads(line)
                if d["idx1"] in ctxs:
                    ctxs[d["idx1"]][d["idx2"]] = d
                else:
                    ctxs[d["idx1"]] = {d["idx2"]: d}

        with open("./musique/musique_train.jsonl") as fin:
            fout = open("./musique/musique_train-ours.json", "w")
            for idx1, line in enumerate(fin):
                d = json.loads(line)
                if not d["answerable"]:
                    continue
                assert len(d["question_decomposition"]) > 1
                for idx2, q in enumerate(d["question_decomposition"]):
                    query = ctxs[idx1][idx2]["query"].strip()
                    assert query.strip() != d["question"].strip()
                    history_qa = [{"Q": ctxs[idx1][idx2_]["query"].strip(), "A": d["question_decomposition"][idx2_]["answer"].strip()} for idx2_ in range(len(d["question_decomposition"])) if idx2_ < idx2]
                    fout.write(json.dumps({
                            "idx1": idx1,
                            "idx2": idx2,
                            "ipt_type": "task_decompose",
                            "opt_type": "next",
                            "ipt": {
                                "task": d["question"].strip(),
                                "query": None,
                                "history": history_qa,
                                "obs": None,
                                },
                            "opt": {"query": query.strip()}
                        })+ "\n")

                    tgt_para = d["paragraphs"][q["paragraph_support_idx"]]

                    allctxs = [{key: ctx[key] for key in ["id", "title", "text"]} for ctx in ctxs[idx1][idx2]["ctxs"] if ctx["title"] != tgt_para["title"]]
                    if len(allctxs):
                        fout.write(json.dumps({
                                "idx1": idx1,
                                "idx2": idx2,
                                "ipt_type": "relevance_judgment",
                                "opt_type": "irrelevant",
                                "ipt": {
                                    "task": d["question"].strip(),
                                    "query": query,
                                    "history": history_qa,
                                    "obs": np.random.choice(allctxs),
                                    },
                                "opt": None,
                                "src": "contriever",
                            })+ "\n")
                        
                    for paraid, para in enumerate(d["paragraphs"]):
                        if para["title"] == tgt_para["title"]:
                            fout.write(json.dumps({
                                    "idx1": idx1,
                                    "idx2": idx2,
                                    "ipt_type": "relevance_judgment",
                                    "opt_type": "relevant",
                                    "ipt": {
                                        "task": d["question"].strip(),
                                        "query": query,
                                        "history": history_qa,
                                        "obs": {"id": None, "title": para["title"], "text": para["paragraph_text"]},
                                        },
                                    "opt": None,
                                    "src": "data",
                                })+ "\n")
                            allctxs = [{key: ctx[key] for key in ["id", "title", "text"]} for ctx in ctxs[idx1][idx2]["ctxs"] if ctx["title"] == tgt_para["title"]]
                            if len(allctxs):
                                fout.write(json.dumps({
                                        "idx1": idx1,
                                        "idx2": idx2,
                                        "ipt_type": "relevance_judgment",
                                        "opt_type": "relevant",
                                        "ipt": {
                                            "task": d["question"].strip(),
                                            "query": query,
                                            "history": history_qa,
                                            "obs": np.random.choice(allctxs),
                                            },
                                        "opt": None,
                                        "src": "contriever",
                                    })+ "\n")

                        if paraid == q["paragraph_support_idx"]:
                            allctxs = [ctx for ctx in ctxs[idx1][idx2]["ctxs"] if ctx["title"] == tgt_para["title"]]
                            assert len(allctxs) <= 1
                            pos_ctxs, neg_ctxs = [], []
                            if len(allctxs):
                                psg_ctxs = allctxs[0]["psg_ctxs"]
                                for ctx in psg_ctxs:
                                    if q["answer"] in ctx["text"]:
                                        pos_ctxs.append(ctx)
                                    else:
                                        neg_ctxs.append(ctx)
                                if len(pos_ctxs) == 1:
                                    para["contriever_text"] = pos_ctxs[0]["text"]
                                    record_id = pos_ctxs[0]["id"]
                                    mix_ctxs = np.random.permutation([pos_ctxs[0]] + neg_ctxs[:2]).tolist()
                                    pos = None
                                    for pos_, ctx in enumerate(mix_ctxs):
                                        if ctx["id"] == record_id:
                                            pos = pos_ + 1
                                            break
                                    fout.write(json.dumps({
                                            "idx1": idx1,
                                            "idx2": idx2,
                                            "ipt_type": "answer_extraction",
                                            "opt_type": "answerable",
                                            "ipt": {
                                                    "task": d["question"].strip(),
                                                    "query": query,
                                                    "history": history_qa,
                                                    "obs": mix_ctxs,
                                                },
                                            "opt": {"ID": pos, "answer": q["answer"]},
                                            "src": "contriever",
                                        })+ "\n")
                                if len(neg_ctxs):
                                    neg_ctxs = np.random.permutation(neg_ctxs).tolist()[:3]
                                    fout.write(json.dumps({
                                            "idx1": idx1,
                                            "idx2": idx2,
                                            "ipt_type": "answer_extraction",
                                            "opt_type": "unanswerable",
                                            "ipt": {
                                                    "task": d["question"].strip(),
                                                    "query": query,
                                                    "history": history_qa,
                                                    "obs": neg_ctxs,
                                                },
                                            "opt": None,
                                            "src": "contriever",
                                        })+ "\n")



                    if idx2 == len(d["question_decomposition"]) - 1:
                        history_qa_finish = [{"Q": ctxs[idx1][idx2_]["query"].strip(), "A": d["question_decomposition"][idx2_]["answer"].strip()} for idx2_ in range(len(d["question_decomposition"]))]
                        fout.write(json.dumps({
                                "ipt_type": "task_decompose",
                                "opt_type": "finish",
                                "ipt": {
                                    "idx1": idx1,
                                    "idx2": None,
                                    "task": d["question"].strip(),
                                    "query": query,
                                    "history": history_qa_finish,
                                    "obs": None,
                                    },
                                "opt": None,
                            })+ "\n")
                        golden_para = []
                        include_contriever = False
                        for idx2_ in range(len(d["question_decomposition"])):
                            golden_para.append(d["paragraphs"][d["question_decomposition"][idx2_]["paragraph_support_idx"]])
                            if "contriever_text" in golden_para[-1]:
                                golden_para[-1]["text"] = golden_para[-1]["contriever_text"]
                                assert d["question_decomposition"][idx2_]["answer"] in golden_para[-1]["contriever_text"]
                                del golden_para[-1]["contriever_text"]
                                include_contriever = True
                            else:
                                golden_para[-1]["text"] = golden_para[-1]["paragraph_text"]
                                assert d["question_decomposition"][idx2_]["answer"] in golden_para[-1]["paragraph_text"]
                            del golden_para[-1]["paragraph_text"]

                        fout.write(json.dumps({
                                "idx1": idx1,
                                "idx2": None,
                                "ipt_type": "finish",
                                "opt_type": "finish",
                                "ipt": {
                                    "task": d["question"].strip(),
                                    "history": history_qa_finish,
                                    "obs": golden_para,
                                },
                                "opt": {"result": d["answer"]},
                                "src": "contriever" if include_contriever else "data",
                            }
                        ) + "\n")

    merge_question()
    retrieve()
    retrievepsg()
    make_train_data_ours()

def pro_2wiki():
    def merge_question():
        with open("./2WikiMultiHopQA/train_pro_sample.json") as fin:
            alld = []
            for line in fin:
                d = json.loads(line)
                alld.append(d)

        for i in range(6):
            with open(f"./2WikiMultiHopQA/train_pro_sample_questions_{i}.json") as fin:
                for line in fin:
                    d = json.loads(line)
                    idx1, idx2 = d["idx1"], d["idx2"]
                    alld[idx1]["evidences"][idx2] = {"triple": alld[idx1]["evidences"][idx2], "_question": d["output"]}

        with open("./2WikiMultiHopQA/train.json", "w") as fout:
            for d in alld:
                fout.write(json.dumps(d) + "\n")


    def retrieve():        
        with open("./2WikiMultiHopQA/train.json") as fin:
            allquestions = []
            for idx1, line in enumerate(fin):
                d = json.loads(line)
                qs = d["evidences"]
                for idx2, q in enumerate(qs):
                    allquestions.append({"idx1": idx1, "idx2": idx2, "query": q["_question"]})
        with open("./2WikiMultiHopQA/train_searchdoc.json", "w") as fout:
            data = {
                "data": allquestions,
                "n_doc": 20,
                "func": "SearchDoc",
            }
            result = retrieve_helper(data)
            for q, r in zip(allquestions, result):
                q["ctxs"] = r["ctxs"]
                fout.write(json.dumps(q) + "\n")

    def retrievepsg():
        with open("./2WikiMultiHopQA/train_searchdoc.json") as fin:
            with open("./2WikiMultiHopQA/train_searchdoc_searchpsg.json", "w") as fout:
                for k, line in enumerate(fin):
                    d = json.loads(line)
                    for c in d["ctxs"]:
                        data = {
                            "data": [{
                                "query": d["query"],
                                "title": c["title"],
                            }],
                            "n_doc": 20,
                            "func": "SearchPsg",
                        }
                        result = retrieve_helper(data)
                        c["psg_ctxs"] = result[0]["ctxs"]
                    fout.write(json.dumps(d) + "\n")


    def make_train_data_ours():
        ctxs = {}
        with open("./2WikiMultiHopQA/train_searchdoc_searchpsg.json") as fin:
            for k, line in enumerate(fin):
                if k % 1000 == 0:
                    print(k)
                d = json.loads(line)
                if d["idx1"] in ctxs:
                    ctxs[d["idx1"]][d["idx2"]] = d
                else:
                    ctxs[d["idx1"]] = {d["idx2"]: d}

        with open("./2WikiMultiHopQA/train.json") as fin:
            fout = open("./2WikiMultiHopQA/train_pro_sample-ours.json", "w")
            for idx1, line in enumerate(fin):
                d = json.loads(line)
                assert "answerable" not in d
                assert len(d["evidences"]) > 1

                for paraid, para in enumerate(d["context"]):
                    para = {'title': para[0], 'paragraph_text': " ".join(para[1])}
                    d["context"][paraid] = para

                for idx2, q in enumerate(d["evidences"]):
                    query = ctxs[idx1][idx2]["query"].strip()
                    assert query.strip() != d["question"].strip()
                    history_qa = [{"Q": ctxs[idx1][idx2_]["query"].strip(), "A": d["evidences"][idx2_]["triple"][-1].strip()} for idx2_ in range(len(d["evidences"])) if idx2_ < idx2]
                    fout.write(json.dumps({
                            "idx1": idx1,
                            "idx2": idx2,
                            "ipt_type": "task_decompose",
                            "opt_type": "next",
                            "ipt": {
                                "task": d["question"].strip(),
                                "query": None,
                                "history": history_qa,
                                "obs": None,
                                },
                            "opt": {"query": query.strip()},
                            "question_type": d["type"],
                        })+ "\n")

                    if len(d["evidences"]) > len(d["supporting_facts"]):
                        continue
                    tgt_para_title = d["supporting_facts"][idx2][0]

                    allctxs = [{key: ctx[key] for key in ["id", "title", "text"]} for ctx in ctxs[idx1][idx2]["ctxs"] if ctx["title"] != tgt_para_title]
                    if len(allctxs):
                        fout.write(json.dumps({
                                "idx1": idx1,
                                "idx2": idx2,
                                "ipt_type": "relevance_judgment",
                                "opt_type": "irrelevant",
                                "ipt": {
                                    "task": d["question"].strip(),
                                    "query": query,
                                    "history": history_qa,
                                    "obs": np.random.choice(allctxs),
                                    },
                                "opt": None,
                                "src": "contriever",
                                "question_type": d["type"],
                            })+ "\n")

                    for paraid, para in enumerate(d["context"]):
                        if para['title'] == tgt_para_title:
                            fout.write(json.dumps({
                                    "idx1": idx1,
                                    "idx2": idx2,
                                    "ipt_type": "relevance_judgment",
                                    "opt_type": "relevant",
                                    "ipt": {
                                        "task": d["question"].strip(),
                                        "query": query,
                                        "history": history_qa,
                                        "obs": {"id": None, "title": para['title'], "text": para['paragraph_text']},
                                        },
                                    "opt": None,
                                    "src": "data",
                                    "question_type": d["type"],
                                })+ "\n")
                            allctxs = [{key: ctx[key] for key in ["id", "title", "text"]} for ctx in ctxs[idx1][idx2]["ctxs"] if ctx["title"] == tgt_para_title]
                            if len(allctxs):
                                fout.write(json.dumps({
                                        "idx1": idx1,
                                        "idx2": idx2,
                                        "ipt_type": "relevance_judgment",
                                        "opt_type": "relevant",
                                        "ipt": {
                                            "task": d["question"].strip(),
                                            "query": query,
                                            "history": history_qa,
                                            "obs": np.random.choice(allctxs),
                                            },
                                        "opt": None,
                                        "src": "contriever",
                                        "question_type": d["type"],
                                    })+ "\n")

                            allctxs = [ctx for ctx in ctxs[idx1][idx2]["ctxs"] if ctx["title"] == tgt_para_title]
                            assert len(allctxs) <= 1
                            pos_ctxs, neg_ctxs = [], []
                            if len(allctxs):
                                psg_ctxs = allctxs[0]["psg_ctxs"]
                                for ctx in psg_ctxs:
                                    if q["triple"][-1] in ctx["text"]:
                                        pos_ctxs.append(ctx)
                                    else:
                                        neg_ctxs.append(ctx)
                                if len(pos_ctxs) == 1:
                                    if "contriever_text" in d["context"][paraid]:
                                        d["context"][paraid]["contriever_text"].append(pos_ctxs[0]["text"])
                                    else:
                                        d["context"][paraid]["contriever_text"] = [pos_ctxs[0]["text"]]
                                    record_id = pos_ctxs[0]["id"]
                                    mix_ctxs = np.random.permutation([pos_ctxs[0]] + neg_ctxs[:2]).tolist()
                                    pos = None
                                    for pos_, ctx in enumerate(mix_ctxs):
                                        if ctx["id"] == record_id:
                                            pos = pos_ + 1
                                            break
                                    fout.write(json.dumps({
                                            "idx1": idx1,
                                            "idx2": idx2,
                                            "ipt_type": "answer_extraction",
                                            "opt_type": "answerable",
                                            "ipt": {
                                                    "task": d["question"].strip(),
                                                    "query": query,
                                                    "history": history_qa,
                                                    "obs": mix_ctxs,
                                                },
                                            "opt": {"ID": pos, "answer": q["triple"][-1].strip()},
                                            "src": "contriever",
                                            "question_type": d["type"],
                                        })+ "\n")
                                if len(neg_ctxs):
                                    neg_ctxs = np.random.permutation(neg_ctxs).tolist()[:3]
                                    fout.write(json.dumps({
                                            "idx1": idx1,
                                            "idx2": idx2,
                                            "ipt_type": "answer_extraction",
                                            "opt_type": "unanswerable",
                                            "ipt": {
                                                    "task": d["question"].strip(),
                                                    "query": query,
                                                    "history": history_qa,
                                                    "obs": neg_ctxs,
                                                },
                                            "opt": None,
                                            "src": "contriever",
                                            "question_type": d["type"],
                                        })+ "\n")



                    if idx2 == len(d["evidences"]) - 1:
                        history_qa_finish = [{"Q": ctxs[idx1][idx2_]["query"].strip(), "A": d["evidences"][idx2_]["triple"][-1].strip()} for idx2_ in range(len(d["evidences"]))]
                        fout.write(json.dumps({
                                "ipt_type": "task_decompose",
                                "opt_type": "finish",
                                "ipt": {
                                    "idx1": idx1,
                                    "idx2": None,
                                    "task": d["question"].strip(),
                                    "query": query,
                                    "history": history_qa_finish,
                                    "obs": None,
                                    },
                                "opt": None,
                                "question_type": d["type"],
                            })+ "\n")
                        golden_para = []
                        include_contriever = False
                        flag = False
                        for idx2_ in range(len(d["evidences"])):
                            golden_idx = None
                            for idx_, para in enumerate(d["context"]):
                                if para["title"] == d["supporting_facts"][idx2_][0]:
                                    golden_idx = idx_
                                    break
                            if "paragraph_text" not in d["context"][golden_idx]:
                                continue
                            golden_para.append(d["context"][golden_idx])

                            ans = d["evidences"][idx2_]["triple"][-1]
                            if "contriever_text" in golden_para[-1]:
                                golden_para[-1]["text"] = None
                                for text_ in golden_para[-1]["contriever_text"]:
                                    if ans in text_:
                                        golden_para[-1]["text"] = text_
                                assert golden_para[-1]["text"] is not None
                                del golden_para[-1]["contriever_text"]
                                include_contriever = True
                            else:
                                golden_para[-1]["text"] = golden_para[-1]["paragraph_text"]
                                try:
                                    assert ans in golden_para[-1]["paragraph_text"]
                                except:
                                    flag = True
                                    break
                            del golden_para[-1]["paragraph_text"]

                        if flag:
                            continue

                        fout.write(json.dumps({
                                "idx1": idx1,
                                "idx2": None,
                                "ipt_type": "finish",
                                "opt_type": "finish",
                                "ipt": {
                                    "task": d["question"].strip(),
                                    "history": history_qa_finish,
                                    "obs": golden_para,
                                },
                                "opt": {"result": d["answer"]},
                                "question_type": d["type"],
                                "src": "contriever" if include_contriever else "data",
                            }
                        ) + "\n")

    merge_question()
    retrieve()
    retrievepsg()
    make_train_data_ours()

def sample_ours():
    with open("./warmup_data.json", "w") as fout:
        for name in [
            "./boolq/train-searchdoc-searchpsg-ours.json",
            "./nq/nq-train-searchdoc-searchpsg-ours.json",
            "./2WikiMultiHopQA/train_pro_sample-ours.json",
            "./musique/musique_train-ours.json",
        ]:
            stat, stat_num = {}, {}
            alld = []
            with open(name) as fin:
                for k, line in enumerate(fin):
                    d = json.loads(line)
                    alld.append(d)
                    ipt_type = d["ipt_type"]
                    opt_type = d["opt_type"]
                    if ipt_type == "task_decompose":
                        if opt_type == "next" and len(d["ipt"]["history"]) >= 2:
                            continue
                        elif opt_type == "finish" and len(d["ipt"]["history"]) >= 3:
                            continue
                    if ipt_type in ["relevance_judgment", "answer_extraction"]:
                        if "src" in d and d["src"] == "data":
                            continue
                        if len(d["ipt"]["history"]) >= 2:
                            continue

                    if ipt_type in ["finish"]:
                        if "src" in d and "musique" in d["src"]:
                            if d["src"] != "contriever":
                                continue
                        if len(d["ipt"]["history"]) >= 3:
                            continue
                    if ipt_type in stat:
                        if opt_type in stat[ipt_type]:
                            stat[ipt_type][opt_type].append(k)
                            stat_num[ipt_type][opt_type] += 1
                        else:
                            stat[ipt_type][opt_type] = [k]
                            stat_num[ipt_type][opt_type] = 1
                    else:
                        stat[ipt_type] = {opt_type: [k]}
                        stat_num[ipt_type] = {opt_type: 1}
            if "2Wiki" in name:
                prob = {'task_decompose': {'next': 3500, 'finish': 500}, 'relevance_judgment': {'irrelevant': 2000, 'relevant': 2000}, 'answer_extraction': {'answerable': 500, 'unanswerable': 500}, 'finish': {'finish': 3000}}
            elif "musique" in name:
                prob = {'task_decompose': {'next': 3500, 'finish': 500}, 'relevance_judgment': {'irrelevant': 2000, 'relevant': 2000}, 'answer_extraction': {'answerable': 3000, 'unanswerable': 1000}, 'finish': {'finish': 4000}}
            elif "nq" in name:
                prob = {'task_decompose': {'next': 500, 'finish': 500}, 'relevance_judgment': {'irrelevant': 2000, 'relevant': 2000}, 'answer_extraction': {'answerable': 1500, 'unanswerable': 1000}, 'finish': {'finish': 1500}}
            elif "boolq" in name:
                prob = {'task_decompose': {'next': 500, 'finish': 500}, 'relevance_judgment': {'irrelevant': 2000, 'relevant': 2000}, 'answer_extraction': {'answerable': 3000, 'unanswerable': 1000}, 'finish': {'finish': 4000}}

            for ipt_type in stat:
                for opt_type in stat[ipt_type]:
                    stat_history_qa = {i:0 for i in range(10)}
                    idx_list = np.random.choice(stat[ipt_type][opt_type], prob[ipt_type][opt_type], replace=False)
                    for idx in idx_list:
                        d = alld[idx]

                        history_qa = d["ipt"]["history"]
                        if history_qa is not None and len(history_qa):
                            history = ["\nSolved Sub-Questions:"]
                            id_ = 0
                            for qa in history_qa:
                                assert qa["A"] is not None
                                if ipt_type == "finish":
                                    if np.random.random() < 0.4:
                                        continue

                                if np.random.random() < 0.2:
                                    history.append(f"{id_+1}. Q: {qa['Q']} A: NO ANSWER")
                                else:
                                    history.append(f"{id_+1}. Q: {qa['Q']} A: {qa['A']}")
                                id_ += 1

                            stat_history_qa[len(history)-1] += 1
                            if len(history) == 1:
                                history = ""
                            else:
                                history = "\n".join(history)
                        else:
                            history = ""
                            stat_history_qa[0] += 1

                        d["ipt"]["history_str"] = history
                        if ipt_type == 'task_decompose':
                            ipt = get_decompose_prompt(task=d["ipt"]["task"], history=history)
                            if opt_type == "next":
                                opt = f"[Next] {d['opt']['query']}"
                            else:
                                assert opt_type == "finish"
                                opt = "[Finish]"
                        elif ipt_type == "relevance_judgment":
                            docs = f'(title: {d["ipt"]["obs"]["title"]}) {d["ipt"]["obs"]["text"]}'
                            ipt = get_relevance_prompt(task=d["ipt"]["task"], history=history, query=d["ipt"]["query"], docs=docs)
                            if opt_type == "irrelevant":
                                opt = "[Irrelevant]"
                            elif opt_type == "relevant":
                                opt = "[Relevant]"
                            else:
                                raise Exception()
                            
                        elif ipt_type == "answer_extraction":
                            docs = "\n".join([f'[{k+1}] (title: {doc["title"]}) {doc["text"]}' for k, doc in enumerate(d["ipt"]["obs"])])
                            ipt = get_solve_prompt(task=d["ipt"]["task"], history=history, query=d["ipt"]["query"], docs=docs)
                            if opt_type == "answerable":
                                opt = f"[Answerable] Answer: {d['opt']['answer']}; Relevant Passage ID: [{d['opt']['ID']}]"
                                tgt_ctx = d["ipt"]["obs"][d['opt']['ID']-1]
                                tgt_text = "(title: %s) %s"%(tgt_ctx["title"], tgt_ctx["text"])
                                if d['opt']['answer'] not in ["yes", "no"]:
                                    assert d['opt']['answer'] in tgt_text
                            else:
                                opt = "[Unanswerable]"
                                tgt_ctx = np.random.choice(d["ipt"]["obs"])
                                tgt_text = "(title: %s) %s"%(tgt_ctx["title"], tgt_ctx["text"])

                        elif ipt_type == "finish":
                            docs = "\n".join([f'[{k+1}] (title: {doc["title"]}) {doc["text"]}' for k, doc in enumerate(d["ipt"]["obs"])])

                            history = ""
                            ipt = get_finish_prompt(task=d["ipt"]["task"], psgs=docs, history=history)
                            opt = d["opt"]["result"]
                        else:
                            raise Exception()

                        ipt_type_2_expert_idx = {
                            "task_decompose": 0,
                            "relevance_judgment": 1,
                            "answer_extraction": 2,
                            "finish": 3,
                        }
                        fout.write(json.dumps({
                            "prompt": ipt,
                            "completion": opt,
                            "expert": ipt_type_2_expert_idx[ipt_type],
                        }) + "\n")

pro_nq()
pro_boolq()
pro_musique()
pro_2wiki()
sample_ours()