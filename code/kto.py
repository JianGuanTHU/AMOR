from dataclasses import dataclass

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from trl import KTOConfig, ModelConfig, get_peft_config
from kto_trainer import KTOTrainer
from modeling_llama import LlamaForCausalLM as AutoModelForCausalLM

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the KTO training script.
    """
    train_dataset_name: str = "./explore/explore_train_reward.json"
    eval_dataset_name: str = "../result/explore_train_reward.json"


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, KTOConfig, ModelConfig))
    script_args, kto_args, model_args = parser.parse_args_into_dataclasses()
    kto_args.max_length = 
    kto_args.desirable_weight = 
    kto_args.undesirable_weight = 
    kto_args.beta = 

    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)
    model_ref = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)
    model.enable_input_require_grads()

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    dataset = load_dataset('json', data_files={'train': script_args.train_dataset_name,'test': script_args.eval_dataset_name})

    def format_dataset(example):
        assert "expert" in example
        example["label"] = bool(example["reward"])
        return example

    formatted_dataset = dataset.map(format_dataset)
    kto_trainer = KTOTrainer(
        model,
        model_ref,
        args=kto_args,
        train_dataset=formatted_dataset["train"],
        eval_dataset=formatted_dataset["test"],
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    kto_trainer.train()
    kto_trainer.save_model(kto_args.output_dir)
    kto_trainer.push_to_hub()
