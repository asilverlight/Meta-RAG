import torch
import argparse
from utils import get_dataset, get_retriever

retrieval_models = {
    'e5':
        {
            'retrieval_method': 'e5',
            'retrieval_model_path': "/your_path/models/E5-base-v2",
            'index_path': '/your_path/wiki_dpr_100w/e5_flat_inner.index',
            'corpus_path': '/your_path/wiki_dpr_100w/wiki_dump.jsonl',
        }
}

config = {
    'data_dir': '/your_path/MetaRAG/results/nq/test.jsonl',
    'index_path': '/your_path/wiki_dpr_100w/e5_flat_inner.index',
    #,# ,'
    'corpus_path': '/your_path/wiki_dpr_100w/wiki_dump.jsonl',
    'retrieval_method': 'e5',#'contriever',#bge
    'retrieval_model_path': "/your_path/models/E5-base-v2",
    'retrieval_topk': 5,
    'save_retrieval_cache': False,
    'use_retrieval_cache': False,
    'retrieval_cache_path': None,
    'retrieval_pooling_method': 'mean',
    'retrieval_query_max_length': 128,
    'retrieval_use_fp16': True,
    'retrieval_batch_size': 256,
    'save_metric_score': True,
    'save_intermediate_data': True,
    'RAG_type': 'meta',
    'batch_size': 5,
    'result_path': '/your_path/multi_llms_for_CoT/results/2wikimultihopqa/checkRAG.jsonl',
    'ensembler':
        {
            'model_name': 'qwen2-7b-instruct',#'qwen2.5-7b-instruct',#'llama2-7B-chat',#
            'model_path': '/your_path/models/Qwen/Qwen2-7B-Instruct',#"/your_path/multi_llms_for_CoT/models/Qwen/Qwen2___5-7B-Instruct",#'/your_path/FlashRAG/models/shakechen/Llama-2-7b-chat-hf',#
            'max_input_len': 8000,
            "framework": "hf",
            'type': torch.bfloat16,
            'stop_token_ids': [151329,
                        151336,
                        151338],
            'gpu_use':0.9,
            'generator_params':
                {
                    'max_tokens': 512,
                    'temperature': 0,
                    'top_p': 0.7,
                },
            'port': '1557'
        }
}
def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--RAG_type', type=str, default='meta')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--dataset_name', type=str, default='nq')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.6)
    parser.add_argument('--refiner', action='store_true')
    parser.add_argument('--model_name', type=str, default='llama3-8B-instruct')
    parser.add_argument('--max_model_len', type=int, default=8192)
    parser.add_argument('--lora', action='store_true')
    parser.add_argument('--lora_path', type=str, default=None)
    parser.add_argument('--max_tokens', type=int, default=1024)
    parser.add_argument('--make_retrieval_results', action='store_true')
    parser.add_argument('--system_prompt', type=int, default=0)
    parser.add_argument('--metares_type', type=str, default='naive')
    parser.add_argument('--exp_type', type=str, default='main')
    parser.add_argument('--model_num', type=int, default=0)
    parser.add_argument('--port', type=str, default='1557')
    parser.add_argument('--retrieval_method', type=str, default='e5')
    parser.add_argument('--data_dir', type=str, default='/your_path/MetaRAG/results/nq/test.jsonl')
    parser.add_argument('--vote_counts', type=int, default=3)
    parser.add_argument('--gpu_num', type=int, default=1)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    args = parser.parse_args()
    config['RAG_type'] = args.RAG_type
    config['batch_size'] = args.batch_size
    config['data_dir'] = '/your_path/MetaRAG/results/' + args.dataset_name + '/retrieval_results.jsonl'
    config['refiner'] = args.refiner
    config['model_name'] = args.model_name
    config['make_retrieval_results'] = args.make_retrieval_results
    config['exp_type'] = args.exp_type
    config['model_num'] = args.model_num
    config['port'] = args.port
    config['retrieval_method'] = args.retrieval_method
    config['retrieval_model_path'] = retrieval_models[args.retrieval_method]['retrieval_model_path']
    config['index_path'] = retrieval_models[args.retrieval_method]['index_path']
    config['corpus_path'] = retrieval_models[args.retrieval_method]['corpus_path']
    config['vote_counts'] = args.vote_counts
    config['gpu_num'] = args.gpu_num
    if args.make_retrieval_results:
        config['data_dir'] = args.data_dir
    
    config['ensembler']['generator_params']['temperature'] = args.temperature
    config['ensembler']['generator_params']['top_p'] = args.top_p
    config['ensembler']['gpu_use'] = args.gpu_memory_utilization
    config['ensembler']['max_input_len'] = args.max_model_len
    config['ensembler']['generator_params']['max_tokens'] = args.max_tokens
    config['ensembler']['generator_params']['repetition_penalty'] = args.repetition_penalty
    config['dataset_name'] = args.dataset_name
    config['lora_path'] = args.lora_path
    config['system_prompt'] = args.system_prompt
    config['metares_type'] = args.metares_type
    test_data = get_dataset(config)
    golden_answers = get_dataset(config, value="golden_answers")
    from pipeline import MetaRAG
    pipeline = MetaRAG(config)
    if config['make_retrieval_results']:
        config['retrieval_topk'] = 5
        config['refiner_topk'] = 5
        config['refiner_encode_max_length'] = 256
        config["refiner_pooling_method"] = 'mean'
        retriever = get_retriever(config)
        datasets = ['nq']#
        config['refiner'] = False
        questions = get_dataset(config)
        pipeline = MetaRAG(config)
        retrieval_results = retriever.batch_search(questions)
        pipeline.save_retrieval_results(questions, retrieval_results)
    else:     
        config['retrieval_topk'] = 8
        config['refiner_topk'] = 3
        config['refiner_encode_max_length'] = 256
        config["refiner_pooling_method"] = 'mean'   
        if args.RAG_type == 'naive':
            pipeline.naive_run(test_data, golden_answers, args.lora)
        elif args.RAG_type == 'meta':
            pipeline.run(test_data, golden_answers, args.lora)
        elif args.RAG_type == 'sc':
            pipeline.self_consistency(test_data, golden_answers, 3, args.lora)
        elif 'logit' in args.RAG_type and 'golden' not in args.RAG_type:
            pipeline.cal_logits(test_data, golden_answers, args.lora)
        elif 'logit' in args.RAG_type and 'golden' in args.RAG_type:
            pipeline.cal_logits_golden_answer(test_data, golden_answers, args.lora)
        elif args.RAG_type == 'best2worst':
            pipeline.best2worst(test_data, golden_answers, args.lora)
        elif args.RAG_type == 'worst2best':
            pipeline.worst2best(test_data, golden_answers, args.lora)
        elif args.RAG_type.lower() == 'sae':
            pipeline.SAE(test_data, golden_answers, args.lora)
        elif args.RAG_type.lower() == 'other_rag':
            pipeline.other_rag(test_data, golden_answers, args.lora)
if __name__ == '__main__':
    run()
