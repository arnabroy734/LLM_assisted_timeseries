from unsloth import FastLanguageModel
import torch
import pandas as pd
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only

def load_model():
    # Model Loading Congif
    base_model_path = 'unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit'
    max_seq_length = 2048
    load_in_4bit = True

    # LORA Config
    r = 16
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj"]
    lora_alpha = 32
    lora_dropout = 0
    bias = "none"
    use_gradient_checkpointing = "unsloth"
    random_state = 3407

    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = base_model_path, 
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        base_model,
        r = r,
        target_modules = target_modules,
        lora_alpha = lora_alpha,
        lora_dropout = lora_dropout,
        bias = bias,
        use_gradient_checkpointing = use_gradient_checkpointing,
        random_state = random_state,
        use_rslora = False,
        loftq_config = None
    )
    return model, tokenizer

def load_data():
    df = pd.read_csv('./llama3_train_health.csv', index_col=0)
    train_data = Dataset.from_pandas(df)
    return train_data

def train_model(model, tokenizer, train_data, max_seq_length):
    train_args = TrainingArguments(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 2,
        warmup_steps = 5,
        num_train_epochs = 10, # Set this for 1 full training run.
        # max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 5,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "tensorboard"
    )
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_data,
        dataset_text_field = "prompt",
        max_seq_length = max_seq_length,
        data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = train_args
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|start_header_id|>user<|end_header_id|>",
        response_part = "<|start_header_id|>assistant<|end_header_id|>",
    )

    print("\n\nStarting training process")
    trainer.train()

    trained_model_path = './lora_model_health'
    model.save_pretrained("lora_model") # Local saving
    tokenizer.save_pretrained("lora_model")
    print(f"Model saved succcessfully")

if __name__=="__main__":
    model, tokenizer = load_model()
    print("\n\nModel and Tokeniser Loaded Successfully")
    train_data = load_data()
    print("\n\nData Loaded Successfully")
    train_model(model, tokenizer, train_data, 2048)





