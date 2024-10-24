import logging
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from modeling_llama import LlamaForCausalLM as AutoModelForCausalLM

class Generator():
    def __init__(self, name=None, args=None):
        self.name = name
        self.args = args
    def Get_Description(self, *args):
        raise NotImplementedError("Get_Description: Not Implemeted")

    def Get_Response(self, *args):
        raise NotImplementedError("Get_Response: Not Implemeted")


class LLama(Generator):
    def __init__(self, name=None, args=None):
        super().__init__(name=name, args=args)
        self.load_model()
    def Get_Description(self):
        return "This is LLama"

    def Get_Response(self, input_text_list, stop=None, module=None):
        if isinstance(input_text_list, str):
            input_text_list = [input_text_list]
        batch_size = self.args.max_batch_size
        response_list = []
        st, ed = 0, 0
        while ed < len(input_text_list):
            st, ed = ed, (ed + batch_size) if (ed + batch_size) < len(input_text_list) else len(input_text_list)
            batch = self.tokenize(input_text_list[st:ed])
            batch = {k: v.to(self.args.device) for k, v in batch.items()}
            
            module2id = {
                "decompose": 0,
                "judge": 1,
                "answer": 2,
                "finish": 3,
            }
            results = self.generator.generate(
                **batch,
                expert_idx=[module2id[module]],
                max_new_tokens=self.args.max_new_tokens,
                do_sample=self.args.do_sample,
                top_p=self.args.top_p,
                temperature=self.args.temperature,
                min_length=0,
                use_cache=True,
                top_k=min([self.args.top_k, len(self.tokenizer)]),
                early_stopping=True,
                num_return_sequences=1,
            )
            _, _, str_outputs = self.decode(batch["input_ids"], results)

            logging.info("="*20)
            logging.info(f"str_outputs: {str_outputs}")

            responses = []
            for pointer in range(len(str_outputs)):
                if stop is not None and stop in str_outputs[pointer]:
                    str_outputs[pointer] = str_outputs[pointer][:str_outputs[pointer].find(stop)]
                responses.append([str_outputs[pointer]])
            response_list += responses
            logging.info(f"response_list: {response_list}")

        return response_list

    def load_model(self):
        print("loading ckpt from %s"%self.args.ckpt_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.ckpt_dir)
        new_tokens = ["<0x0%d>"%d for d in range(10)]
        self.tokenizer.add_tokens(new_tokens, special_tokens=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.generator = AutoModelForCausalLM.from_pretrained(self.args.ckpt_dir).to(self.args.device)

    def tokenize(self, dialogs):
        input_ids = self.tokenizer(dialogs)
        tokenize_input_ids = input_ids["input_ids"]
        pad_input_ids, pad_attention_mask = [], []
        maxlen = min([max([len(input_ids) for input_ids in tokenize_input_ids]), self.args.max_input_tokens])
        for input_ids in tokenize_input_ids:
            iptlen = len(input_ids)
            if iptlen < maxlen:
                attention_mask = [0 for _ in range(maxlen-iptlen)] + [1 for _ in range(iptlen)]
                input_ids = [2 for _ in range(maxlen-iptlen)] + input_ids
            else:
                input_ids = input_ids[-maxlen:]
                attention_mask = [1 for _ in range(maxlen)]
            pad_input_ids.append(input_ids)
            pad_attention_mask.append(attention_mask)
        return {"input_ids": torch.tensor(pad_input_ids), "attention_mask": torch.tensor(pad_attention_mask)}

    def decode(self, prompts, samples):
        prompt_sizes = [prompts.shape[1]] * len(prompts)

        str_samples, str_prompts, str_outputs = [], [], []
        for prompt, sample, prompt_size in zip(prompts, samples, prompt_sizes):
            output_start_ix = prompt_size
            str_prompt = self.tokenizer.decode(prompt[:prompt_size], skip_special_tokens=True)
            str_output = self.tokenizer.decode(sample[output_start_ix:], skip_special_tokens=True)
            logging.info(f"sample: {self.tokenizer.decode(sample, skip_special_tokens=True)}")
            str_prompts.append(str_prompt)
            str_outputs.append(str_output)

            sample = str_prompt + str_output
            str_samples.append(sample)

        return str_samples, str_prompts, str_outputs
