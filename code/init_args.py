from dataclasses import dataclass, field

@dataclass
class Arguments:
    name: str = field(default="", metadata={"help": ""})

@dataclass
class GeneratorArguments(Arguments):
    model: str = field(default="llama")
    ckpt_dir: str = field(default="./model/warmup_model", metadata={"help": ""})    
    device: str = field(default="cuda:0", metadata={"help": ""})

    max_batch_size: int = field(default=4, metadata={"help": ""})
    max_input_tokens: int = field(default=3584, metadata={"help": ""})
    max_new_tokens: int = field(default=512, metadata={"help": ""})
    do_sample: bool = field(default=False, metadata={"help": ""})
    top_p: float = field(default=1.0, metadata={"help": ""})
    top_k: int = field(default=100000, metadata={"help": ""})
    temperature: float = field(default=1.0, metadata={"help": ""})

@dataclass
class ManagerArguments(Arguments):
    log: str = field(default=None, metadata={"help": ""})
    generator_args: GeneratorArguments = field(default=GeneratorArguments(), metadata={"help": ""})
