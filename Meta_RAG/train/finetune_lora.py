from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model, PeftModel
import torch
import wandb
import argparse
import random
from accelerate import Accelerator
from typing import Optional
from fine_tune_qa_dataset import FineTuningQADataset, FineTuningMetaRAG
import os
import ujson
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=2)

def tokenize_batch_for_finetune(batch, tokenizer: Optional[AutoTokenizer] = None, max_length: int = 1024, rag_type='naive', model_name=None, prompt_num=0):
    texts = []
    for sample in batch:
        if rag_type == 'naive':
            chat_prompt = [
                {
                    "role": "system",
                    "content": f"Answer the question based on the given document. Only give me the answer and do not output any other words.\nThe following are given documents.\n\n{sample['reference']}"
                },
                {
                    "role": "user",
                    # "content": f'{sample["prompt"]}'
                    "content": f"Question: {sample['question']}"
                },
            ]
        else:
            chat_prompt = [
                    {
                        "role": "system",
                        "content": f"""
You are an exceptional text analysis assistant. I will provide you with a question and some information from {sample['num']} different systems. Each system's information includes external reference data and a corresponding answer derived from the data. Your task is to comprehensively analyze all the given information. Analyze the content of all reference documents. Use the content from all sets to synthesize a single, final answer. Follow these requirements strictly:

1. First, output your reasoning process briefly, shortly and concretely, including how you evaluated the information from each system on the first line.
2. Then, output your final answer based on your analysis on the first line.
3. Do not include anything other than your reasoning process and the final answer.\n\n
"""
                    },
                    {
                        "role": "user",
                        "content": f"Question: {sample['question']}"
                    },
                ]
            if prompt_num == 0:
                chat_prompt[0]['content'] = f"""Answer the question based on the given external data from {sample['num']} systems' information. Each system's information contains reference documents and an answer derived from those documents. Analyze the content of all reference documents. Use the content from all sets to synthesize a single, final answer. Only give me the final answer and do not output any other words. The following are the given external data sets:\n\n{sample['reference']}"""
            elif prompt_num == 1:
                chat_prompt[0]['content'] = f"""Answer the question based on the given external data from {sample['num']} systems' information. Each system's information contains reference documents and an answer derived from those documents. The correctness of each system's data may vary. Analyze the content of all reference documents. Use the content from all sets to synthesize a single, final answer. Only give me the final answer and do not output any other words. The following are the given external data sets:\n\n{sample['reference']}"""
            elif prompt_num == 2:
                chat_prompt[0]['content'] = f"""Answer the question based on the given external data from {sample['num']} systems' information. Each system's information contains reference documents and an answer derived from those documents. The correctness of each system's data may vary, and the ability of each system to answer different types of questions varies. Analyze the content of all reference documents. Use the content from all sets to synthesize a single, final answer. Only give me the final answer and do not output any other words. The following are the given external data sets:\n\n{sample['reference']}"""
            elif prompt_num == 3:
                chat_prompt[0]['content'] = f"""Determine whether the fact in the question is correct based on the given external data from {sample['num']} systems' information. Each system's information contains reference documents and a judgment derived from those documents. Use the content from all sets to synthesize a single, final judgment. If the question is correct, output 'SUPPORTS', otherwise output 'REFUTES' due to the fact contradiction. Onlu give me a 'SUPPORTS' or 'REFUTES', and do not output any other words. The following are the given external data sets:\n\n{sample['reference']}"""
            elif prompt_num == 4:
                chat_prompt[0]['content'] = f"""Write a detailed and coherent response to the following question based on the given external data from {sample['num']} systems' information. Each system's information contains reference documents and an answer derived from those documents. Use the content from all sets to synthesize a single, final answer. Only give me the final answer and do not output any other words. The following are the given external data sets:\n\n{sample['reference']}"""
            elif prompt_num == 5:
                chat_prompt[0]['content'] = f"""
Answer the question based on the given external data from {sample['num']} systems' information. Each system's information contains reference documents and an answer derived from those documents. Here is things to pay attention to:
- First analyze the content of all reference documents.
- Then provide step-by-step reasoning on how to answer the question.
- In the reasoning, if you need to copy paste some sentences from the context, include them in ##begin_quote## and ##end_quote##. This would mean that things outside of ##begin_quote## and ##end_quote## are not directly copy paste from the context.
- End your response with the final answer, and the answer should be succinct, such as a few words, but not a complete sentence.
- You should return your output in JSON format as follows:
  {{\"reason\": <your reason>, \"answer\": <your answer>}}
Strictly follow the required format and content output, and do not output any other words.
"""
                chat_prompt[1]['content'] = f"Question: {sample['question']}\n\nSystems' information: \n{sample['reference']}\n\n"
            # chat_prompt.append({
            #     "role": "assistant",
            #     "content": f"{sample['completion']}"
            # })
        texts.append(chat_prompt)
    # tokenizer.padding_side = "left"
    # tokenizer.truncation_side = "left"
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token
    if 'llama' in model_name.lower():
        texts = [tokenizer.apply_chat_template(chat_prompt, tokenize=False, add_generation_prompt=True) for chat_prompt in texts]
        texts = [text + sample["completion"] + tokenizer.eos_token for text, sample in zip(texts, batch)]
        completion = [sample["completion"] + tokenizer.eos_token for sample in batch]
    elif 'qwen' in model_name.lower():
        texts = [
            f"<|im_start|>system\n{chat_prompt[0]['content']}<|im_end|>\n<|im_start|>user\n{chat_prompt[1]['content']}<|im_end|>\n<|im_start|>assistant\n{sample['completion']}<|im_end|>"
            for chat_prompt, sample in zip(texts, batch)
        ]
        completion = [sample["completion"] + '<|im_end|>' for sample in batch]
    elif 'mistral' in model_name.lower():
        texts = [tokenizer.apply_chat_template([
            {
                "role": "user",
                "content": f"{chat_prompt[0]['content']}\n{chat_prompt[1]['content']}"
            },
            {
                "role": "assistant",
                "content": f"{sample['completion']}"
            }
            ], tokenize=False, add_generation_prompt=True) for chat_prompt, sample in zip(texts, batch)]
        completion = [sample["completion"] + tokenizer.eos_token for sample in batch]
    elif 'glm' in model_name.lower():
        texts = [
            f"[gMASK]<sop><|system|>\n{chat_prompt[0]['content']}<|user|>\n{chat_prompt[1]['content']}<|assistant|>\n{sample['completion']}<|endoftext|>"
            for chat_prompt, sample in zip(texts, batch)
        ]
        completion = [sample["completion"] + '<|endoftext|>' for sample in batch]
    elif 'gemma' in model_name.lower():
        texts = [
            f"<bos><start_of_turn>user\n{chat_prompt[0]['content']}\n{chat_prompt[1]['content']}<end_of_turn>\n<start_of_turn>model\n{sample['completion']}<end_of_turn>\n"
            for chat_prompt, sample in zip(texts, batch)
        ]
        completion = [sample["completion"] + '<end_of_turn>\n' for sample in batch]
    else:
        raise ValueError("Unsupported model name")
    data = tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length, add_special_tokens=True)
    data_completion = tokenizer(completion, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length, add_special_tokens=False)
    data_mask_reverse = 1 - data_completion["attention_mask"]
    data_mask = data_mask_reverse * -100
    data["labels"] = data["input_ids"].clone()
    data["labels"] *= data_completion["attention_mask"]
    data["labels"] += data_mask
    # data["input_ids"].requires_grad = True
    data = {k: v.cuda() for k, v in data.items()}
    return data

def run_ft(args):
    torch.set_grad_enabled(True)
    accelerator = Accelerator()#device_placement=False
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    # if 'llama' in args.model_path.lower() or 'mistral' in args.model_path.lower():
    #     tokenizer = set_tokenizer(tokenizer)
    target_modules = ["q_proj", "v_proj"]
    peft_config = LoraConfig(
        r=8, 
        lora_alpha=32, 
        inference_mode=False,
        lora_dropout=0.05, 
        bias="none", 
        task_type="CAUSAL_LM"
    )

    if args.resume_from_checkpoint is not None:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="flash_attention_2")#
        model = PeftModel.from_pretrained(model, args.resume_from_checkpoint)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="flash_attention_2")#
        model = get_peft_model(model, peft_config)

    model.to(accelerator.device)
    model.enable_input_require_grads()
    # model.print_trainable_parameters()
    # assert False

    training_args = TrainingArguments(
        weight_decay=0.01,
        output_dir=args.output_path,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epoch,
        per_device_train_batch_size=args.per_device_train_batch_size,
        dataloader_pin_memory=False,
        do_eval=False,
        save_total_limit=6,
        warmup_ratio=0.02,
        save_strategy="epoch",
        logging_steps=10,
        bf16=True,
        # label_names=["completion"],
        report_to="wandb",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_accumulation_steps=8,
    )
    model_path_names = {
        "/your_path/models/ZhipuAI/glm-4-9b-chat": 
            "glm-4-9b-chat",
        "/your_path/models/LLM-Research/Mistral-7B-Instruct-v0___3":
            "mistral-7B-instruct-v0.3",
        '/your_path/models/Llama-2-7b-chat-hf':
            "llama2-7B-chat",
        '/your_path/models/Qwen/Qwen2-7B-Instruct':
            "qwen2-7b-instruct",
        '/your_path/models/LLM-Research/Meta-Llama-3-8B-Instruct':
            "llama3-8B-instruct",  
        '/your_path/Qwen2.5-7B-Instruct':
            'qwen2.5-7b-instruct',
        '/your_path/Llama-3.1-8B-Instruct':
            'llama3.1-8B-instruct',
        '/your_path/LLM-Research/Llama-3___2-1B-Instruct':
            'llama3.2-1B-instruct',
        '/your_path/LLM-Research/Llama-3___2-3B-Instruct':
            'llama3.2-3B-instruct',
        '/your_path/Qwen/Qwen2___5-1___5B-Instruct':
            'qwen2.5-1.5b-instruct',
        '/your_path/Qwen/Qwen2___5-0___5B-Instruct':
            'qwen2.5-0.5b-instruct',
        '/your_path/Qwen/Qwen2___5-3B-Instruct':
            'qwen2.5-3b-instruct',
        '/your_path/LLM-Research/OpenELM-3B-Instruct':
            'openelm-3b-instruct',
        '/your_path/Qwen/Qwen2___5-3B':
            'qwen2.5-3b',
        '/your_path/LLM-Research/gemma-2-2b-it':
            'gemma-2-2b-it',
        '/your_path/AI-ModelScope/gemma-2b-it':
            'gemma-2b-it',
    }
    prompt_num = args.prompt_num
    if 'fever' in args.dataset:
        prompt_num = 3
    elif 'eli5' in args.dataset or 'wikiasp' in args.dataset:
        prompt_num = 4
    model_name = model_path_names[args.model_path.rstrip('/')]
    if args.rag_type == 'naive':
        train_ds = FineTuningQADataset(args.dataset, with_ret=True, zero_shot=True, ret_passages=args.ret_passages, model_name=model_name)
    else:
        train_ds = FineTuningMetaRAG(args.dataset, with_ret=True, zero_shot=False, ret_passages=args.ret_passages, model_name=model_name, refiner=args.refiner, prompt_num=prompt_num)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=lambda data: tokenize_batch_for_finetune(data, tokenizer=tokenizer, max_length=args.max_length, rag_type=args.rag_type, model_name=model_name, prompt_num=prompt_num),
        # callbacks=[early_stopping_callback],
    )
    # torch.set_grad_enabled(True)

    trainer.train()
    trainer.save_model()
    trainer.save_state()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--max_length", type=int, default=600)
    parser.add_argument("--ret_passages", type=int, default=5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--rag_type", type=str, default="naive")
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--refiner", action="store_true")
    parser.add_argument('--prompt_num', type=int, default=0)
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Path to checkpoint to resume training from')
    args = parser.parse_args()

    wandb.init(mode="disabled")
    import sys
    log_file = args.output_path + "/log.txt"
    sys.stdout = open(log_file, "w")
    run_ft(args)
    sys.stdout.close()
    sys.stdout = sys.__stdout__

if __name__ == "__main__":
    main()