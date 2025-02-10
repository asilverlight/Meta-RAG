export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=7
accelerate launch --config_file /your_path/MetaRAG/scripts/train/zero1_gpu.yml --main_process_port=1152 /your_path/MetaRAG/scripts/train/finetune_lora.py \
  --model_path /your_path/LLM-Research/Llama-3___2-1B-Instruct/ \
  --output_path /your_path/models/lora/ \
  --tokenizer_path /your_path/LLM-Research/Llama-3___2-1B-Instruct/ \
  --dataset /your_path/MetaRAG/scripts/train/meta_train.jsonl \
  --max_length 5000 \
  --rag_type meta \
  --per_device_train_batch_size 2 \
  --epoch 3 \
