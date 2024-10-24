import json
import os
import numpy as np
np.random.seed(42)
from run import get_metrics
from prompts import get_decompose_prompt, get_relevance_prompt, get_solve_prompt, get_finish_prompt

def get_history_str(history_qa):
    if len(history_qa):
        history = ["\nSolved Sub-Questions:"]
        id_ = 0
        for qa in history_qa:
            if qa[1] is None:
                history.append(f"{id_+1}. Q: {qa[0]} A: NO ANSWER")
            else:
                history.append(f"{id_+1}. Q: {qa[0]} A: {qa[1]}")
            id_ += 1

        if len(history) == 1:
            history = ""
        else:
            history = "\n".join(history)
    else:
        history = ""
    return history

def get_process_feedback(original_file, explore_file):
    with open(f"{original_file}") as fin:
        alldata = {}
        for line in fin:
            d = json.loads(line)
            alldata[d["_id"]] = d
    ipt_type_2_expert_idx = {
        "task_decompose": 0,
        "relevance_judgment": 1,
        "answer_extraction": 2,
        "finish": 3,
    }
    with open(explore_file.replace(".json", "_reward.json"), "w") as fout:
        with open(f"{explore_file}") as fin:
            for line in fin:
                d = json.loads(line)
                traj = d["traj"]

                metrics = get_metrics(pred=d["pred"], truth=d["answer"])
                em = metrics["em"]

                e1 = alldata[d["_id"]]["supporting_facts"][0][0]
                e2 = alldata[d["_id"]]["supporting_facts"][1][0]

                e1_ctx = alldata[d["_id"]]["supporting_facts"][0][3]
                e2_ctx = alldata[d["_id"]]["supporting_facts"][1][3]
                task = d["question"]
                all_record = []

                if traj is None:
                    continue
                solve_decompose_idx, e1_, e2_, pass_solve_decompose = -1, 0, 0, [False, False, False]
                for idx_, traj_ in enumerate(traj):
                    if traj_["type"] == "solve_decompose":
                        solve_decompose_idx += 1
                    if "state" in traj_ and traj_["state"]["obs"] is not None:
                        if f"(title: {e1})" in traj_["state"]["obs"]:
                            e1_ = 1
                            pass_solve_decompose[solve_decompose_idx] = True
                        if f"(title: {e2})" in traj_["state"]["obs"]:
                            e2_ = 1
                            pass_solve_decompose[solve_decompose_idx] = True
                for traj_idx_, traj_ in enumerate(traj):
                    history_qa = traj_["state"]["history_qa"]
                    history = get_history_str(history_qa=history_qa[:-1])
                    history_full = get_history_str(history_qa=history_qa)

                    if traj_["type"] == "solve_decompose":
                        ipt_type = "task_decompose"
                        if traj[traj_idx_+1]["type"] == "solve_searchdoc":
                            opt_type = "next%d"%(len(traj_["state"]["history_qa"])-1)
                            opt = f"""[Next] {traj_["state"]["history_qa"][-1][0].strip()}"""
                            tmp_history = history
                        elif traj[traj_idx_+1]["type"] == "solve_finish":
                            opt_type = "finish"
                            opt = "[Finish]"
                            tmp_history = history_full
                        else:
                            raise Exception()
                        ipt = get_decompose_prompt(task=task, history=tmp_history)

                        if opt_type == "next0" or opt_type == "next1":
                            if pass_solve_decompose[int(opt_type[-1])]:
                                feedback = "right"
                            else:
                                feedback = "wrong"
                        elif opt_type == "finish":
                            feedback = "right"
                        else:
                            raise Exception()
                    elif traj_["type"] == "solve_doc_relevance":
                        ipt_type = "relevance_judgment"
                        docs = traj_["state"]["obs"]
                        query = traj_["state"]["history_qa"][-1][0].strip()
                        if traj[traj_idx_+1]["type"] in ["solve_next", "solve_decompose"]:
                            opt_type = "irrelevant"
                            opt = "[Irrelevant]"
                        elif traj[traj_idx_+1]["type"] == "solve_search_psg":
                            opt_type = "relevant"
                            opt = "[Relevant]"
                        else:
                            raise Exception()
                        ipt = get_relevance_prompt(task=task, history=history, query=query, docs=docs)

                        truth_opt = "[Irrelevant]"
                        for ee in [e1, e2]:
                            if f"(title: {ee})" in docs:
                                truth_opt = "[Relevant]"
                                break

                        opt = truth_opt
                        feedback = "right"
                        if len(history):
                            if not pass_solve_decompose[1]:
                                opt = opt
                                feedback = "skip"
                        else:
                            if not pass_solve_decompose[0]:
                                opt = opt
                                feedback = "skip"

                    elif traj_["type"] == "solve_psg_relevance":
                        ipt_type = "answer_extraction"
                        docs = [mem[3:].strip() for mem in traj_["state"]["obs"].strip().split("\n")]
                        query = traj_["state"]["history_qa"][-1][0].strip()
                        ipt = get_solve_prompt(task=task, history=history, query=query, docs="\n".join([f'[{kk+1}] {tmpdoc}' for kk, tmpdoc in enumerate(docs)]))
                        ans = None
                        if traj[traj_idx_+1]["type"] == "solve_decompose":
                            opt_type = "answerable"
                            memobs = [mem[3:].strip() for mem in traj[-1]["state"]["obs"].strip().split("\n")]

                            ans_psg, ans_psg_id = None, None
                            for mem in memobs:
                                flag = False
                                for _id_, doc in enumerate(docs):
                                    if doc in mem:
                                        ans_psg_id = _id_+1
                                        ans_psg = doc
                                        flag = True
                                        break
                                if flag:
                                    break

                            if traj_["state"]["history_qa"][-1][1] is None:
                                opt_type = "unanswerable"
                                tgt_text = np.random.choice(docs)
                                opt = "[Unanswerable]"
                            else:
                                ans = traj_["state"]["history_qa"][-1][1].strip()
                                opt = f"[Answerable] Answer: {ans}; Relevant Passage ID: [{ans_psg_id}]"
                        elif traj[traj_idx_+1]["type"] == "solve_next":
                            opt_type = "unanswerable"
                            tgt_text = np.random.choice(docs)
                            opt = "[Unanswerable]"
                        else:
                            raise Exception()

                        truth_opt = "[Unanswerable]"
                        for ee_ctx in [e1_ctx, e2_ctx]:
                            if ee_ctx is not None:
                                tgt_text = f"(title: {ee_ctx['title']}) {ee_ctx['text']}"
                                if tgt_text in docs:
                                    pos_id = docs.index(tgt_text) + 1
                                    truth_opt = f"[Answerable] Answer: {ans}; Relevant Passage ID: [{pos_id}]"
                                    break

                        feedback = "right" if opt==truth_opt else "wrong"

                        if len(history):
                            if not pass_solve_decompose[1]:
                                feedback = "skip"
                        else:
                            if not pass_solve_decompose[0]:
                                feedback = "skip"

                    elif traj_["type"] == "solve_finish":
                        ipt_type = "finish"
                        opt_type = "finish"
                        docs = traj_["state"]["obs"].strip()

                        ipt = get_finish_prompt(task=task, psgs=docs, history="")
                        opt = d["pred"]
                        if em == 1:
                            feedback = "right"
                        else:
                            feedback = "wrong"
                    else:
                        continue

                    if feedback != "skip":
                        all_record.append({
                            "ipt_type": ipt_type,
                            "opt_type": opt_type,
                            "prompt": ipt,
                            "completion": opt,
                            "reward": 1 if feedback=="right" else 0,
                            "expert": ipt_type_2_expert_idx[ipt_type],
                        })

                if em != 1:
                    traj_ = d["traj"][-1]
                    e1_ = 1 if f"(title: {e1})" in traj_["state"]["obs"] else 0
                    e2_ = 1 if f"(title: {e2})" in traj_["state"]["obs"] else 0
                    answer_recall = int(get_metrics(pred=traj_["state"]["obs"], truth=d["answer"])["recall"]==1)

                    ipt_type = "finish"
                    opt_type = "finish"
                    docs = traj_["state"]["obs"].strip()

                    ipt = get_finish_prompt(task=task, psgs=docs, history="")
                    opt = d["answer"]

                    if (answer_recall or (e1_ == 1 and e2_ == 1)):
                        feedback = "right"
                    else:
                        feedback = "wrong"
                    all_record.append({
                        "prompt": ipt,
                        "completion": opt,
                        "expert": ipt_type_2_expert_idx[ipt_type],
                        "reward": 1 if feedback=="right" else 0,
                    })
                for record in all_record:
                    fout.write(json.dumps(record) + "\n")

get_process_feedback(original_file=f"../data/hotpotqa/train.json", explore_file=f"../result/explore_train.json")