import copy
from prompts import get_decompose_prompt, get_relevance_prompt, get_solve_prompt, get_finish_prompt
import logging
from generator import LLama
import texttools

class State():
    def __init__(self, task=None, history_qa=None, obs=None):
        self.task = task
        self.history_qa = history_qa
        self.obs = obs
        self.doc = None


class Manager():
    def __init__(self, args=None):
        if args is not None:
            self.generator_args = args.generator_args
            self.generator = LLama(args=self.generator_args)
        self.text_toolkits = texttools.text_toolkits()
        self.max_turn = 2

    def initialize(self):
        self.docs = []
        self.doc_pointer = None

        self.memory = []
        self.traj = []

    def record_traj(self, type, state):
        self.traj.append({"type": type, "state": {"task": state.task, "history_qa": state.history_qa, "obs": state.obs}})

    def solve_decompose(self, state):
        task = state.task.strip()
        history_qa = state.history_qa
        state.obs = None
        state.doc = None

        if len(history_qa) >= self.max_turn:
            self.record_traj(type="solve_decompose", state=state)
            return self.solve_finish(state)

        if len(history_qa):
            history = "\nSolved Sub-Questions:\n%s"%("\n".join([f"{id_+1}. Q: {qa[0]} A: {qa[1] if qa[1] is not None else 'NO ANSWER'}" for id_, qa in enumerate(history_qa)]))
        else:
            history = ""

        prompt = get_decompose_prompt(task=task, history=history)
        logging.info(f"solve_decompose PROMPT: {len(self.generator.tokenizer.tokenize(prompt))} {prompt}")
        output = self.generator.Get_Response(input_text_list=[prompt], stop="\n", module="decompose")[0][0]
        logging.info(f"solve_decompose OUTPUT: {output}")
        output = output.strip().split("\n")[0].strip()

        if "[Next]" in output:
            next_query = output.strip().split("[Next]")[1].strip()
            state.history_qa = state.history_qa + [[next_query, None]]
            self.record_traj(type="solve_decompose", state=state)
            return self.solve_searchdoc(state)
        elif "[Finish]" in output:
            self.record_traj(type="solve_decompose", state=state)
            return self.solve_finish(state)
        else:
            raise Exception("solve_decompose ERROR")

    def solve_finish(self, state, example_str=None):
        task = state.task.strip()
        mems = []
        for mem in self.memory:
            if mem not in mems:
                mems.append(mem)
        obs = "\n".join([f"[{k+1}] {mem.strip()}" for k, mem in enumerate(mems)])
        state.obs = obs
        history_qa = state.history_qa
        history_qa_list = [f"{id_+1}. Q: {qa[0]} A: {qa[1]}" for id_, qa in enumerate(history_qa) if qa[1] is not None]
        if len(history_qa_list):
            history = "\nDecomposed Question-Answering Pairs:\n%s"%("\n".join(history_qa_list))
        else:
            history = ""

        history = ""
        prompt = get_finish_prompt(task=task, psgs=obs, history=history)
        logging.info(f"solve_finish PROMPT: {len(self.generator.tokenizer.tokenize(prompt))} {prompt}")
        output = self.generator.Get_Response(input_text_list=[prompt], stop=None, module="finish")[0][0]
        output = output.strip().split("\n")[0].strip()
        logging.info(f"solve_finish OUTPUT: {output}")
        output = output.strip()
        self.record_traj(type="solve_finish", state=state)
        return output

    def solve_searchdoc(self, state):
        # search document
        query = state.history_qa[-1][0].strip()
        ctxs = self.text_toolkits.SearchDoc(query=query)
        state.obs = f'(title: {ctxs[0]["title"].strip()}) {ctxs[0]["text"].strip()}'
        state.doc = ctxs[0]

        self.docs = copy.deepcopy(ctxs)
        self.doc_pointer = 0

        self.record_traj(type="solve_searchdoc", state=state)
        return self.solve_doc_relevance(state)

    def solve_doc_relevance(self, state):
        type_name = "solve_doc_relevance"
        # judge relevant
        task = state.task.strip()
        history_qa = state.history_qa
        query = history_qa[-1][0].strip()
        obs = state.obs.strip()
        if len(history_qa[:-1]):
            history = "\nSolved Sub-Questions:\n%s"%("\n".join([f"{id_+1}. Q: {qa[0]} A: {qa[1] if qa[1] is not None else 'NO ANSWER'}" for id_, qa in enumerate(history_qa[:-1])]))
        else:
            history = ""

        prompt = get_relevance_prompt(task=task, query=query, docs=obs, history=history)
        logging.info(f"{type_name} PROMPT: {len(self.generator.tokenizer.tokenize(prompt))} {prompt}")
        output = self.generator.Get_Response(input_text_list=[prompt], stop="\n", module="judge")[0][0]
        logging.info(f"{type_name} OUTPUT: {output}")
        output = output.strip().split("\n")[0].strip()
        if "[Relevant]" in output:
            self.record_traj(type=type_name, state=state)
            return self.solve_search_psg(state)
        else:
            self.record_traj(type=type_name, state=state)
            return self.solve_next(state)

    def solve_next(self, state):
        self.doc_pointer += 1
        if self.doc_pointer == len(self.docs):
            self.memory += [f'(title: {self.docs[0]["title"].strip()}) {self.docs[0]["text"].strip()}']
            return self.solve_decompose(state)
        else:
            obs = f'(title: {self.docs[self.doc_pointer]["title"].strip()}) {self.docs[self.doc_pointer]["text"].strip()}'
            state.obs = obs.strip()
            state.doc = self.docs[self.doc_pointer]
            self.record_traj(type="solve_next", state=state)
            return self.solve_doc_relevance(state)

    def solve_search_psg(self, state):
        query = state.history_qa[-1][0].strip()
        obs = self.text_toolkits.SearchPsg(query=query, title=state.doc["title"])
        obs = "\n".join([f'[{idx_+1}] (title: {ctx["title"].strip()}) {ctx["text"].strip()}'
                            for idx_, ctx in enumerate(obs[:3]) #1:4
                        ])
        state.obs = obs
        self.record_traj(type="solve_search_psg", state=state)
        return self.solve_psg_relevance(state)

    def solve_psg_relevance(self, state):
        type_name = "solve_psg_relevance"
        task = state.task.strip()
        history_qa = state.history_qa
        query = history_qa[-1][0].strip()
        obs = state.obs.strip()
        if len(history_qa[:-1]):
            history = "\nSolved Sub-Questions:\n%s"%("\n".join([f"{id_+1}. Q: {qa[0]} A: {qa[1] if qa[1] is not None else 'NO ANSWER'}" for id_, qa in enumerate(history_qa[:-1])]))
        else:
            history = ""

        prompt = get_solve_prompt(task=task, query=query, docs=obs, history=history)
        logging.info(f"{type_name} PROMPT: {len(self.generator.tokenizer.tokenize(prompt))} {prompt}")
        output = self.generator.Get_Response(input_text_list=[prompt], stop="\n", module="answer")[0][0]
        logging.info(f"{type_name} OUTPUT: {output}")
        output = output.strip().split("\n")[0].strip()
        if "Answerable]" in output:
            try:
                ans_id = output.strip().split("Answerable] Answer:")[1].strip()
                if "; Relevant Passage ID:" in ans_id:
                    ans, id_ = tuple(ans_id.strip().split("; Relevant Passage ID:"))
                else:
                    ans, id_ = tuple(ans_id.strip().split("Relevant Passage ID:"))
            except:
                self.record_traj(type=type_name, state=state)
                return self.solve_next(state)
            try:
                idx = eval(id_.strip())[0]
            except:
                try:
                    idx = eval(id_.strip()[1])
                except:
                    idx = 1

            if isinstance(idx, list) and len(idx):
                idx = idx[0]
            mem = None
            obs_lines, st, pointer = [], 0, 1
            while True:
                if obs[st:].startswith(f"[{pointer}] (title:"):
                    if f"[{pointer+1}] (title:" in obs[st:]:
                        ed = obs[st:].find(f"[{pointer+1}] (title:")
                        obs_lines.append(obs[st:ed+st].strip())
                        st = ed+st
                        pointer += 1
                    else:
                        obs_lines.append(obs[st:].strip())
                        break
                else:
                    raise Exception()
            for obs_line in obs_lines:
                if obs_line.startswith(f"[{idx}]"):
                    mem = obs_line.strip()[3:].strip()
                    break
            if mem is not None:
                self.memory += [mem]

            state.history_qa = state.history_qa[:-1] + [[state.history_qa[-1][0], ans.strip()]]
            self.record_traj(type=type_name, state=state)
            return self.solve_decompose(state)
        else:
            self.record_traj(type=type_name, state=state)
            return self.solve_next(state)

    def Solve_Executor(self, task):
        self.initialize()
        s0 = State(task=task.strip(), history_qa=[], obs=None)
        result = self.solve_decompose(s0)
        
        return result, self.traj
