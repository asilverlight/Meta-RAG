import time
import copy
from tqdm import tqdm
from utils import retain_after_last_substring
import asyncio
import re
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
import gc
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForCausalLM
import ujson
from utils import get_retriever
import string
import numpy as np
import random
from transformers import FalconConfig
import torch.nn.functional as F
rng = np.random.default_rng(1557)

class BasicPipeline:
    def __init__(self, config):
        self.config = config
        self.batch_size = config['batch_size']
        self.result_path = ''
        self.result_path_indent4 = ''
        self.rag_type = config['RAG_type']
        
    def run(self):
        pass

        
class MetaRAG(BasicPipeline):
    def __init__(self, config, retriever=None):
        super().__init__(config)   
        self.model_names = ['llama2-7B-chat', 'qwen2-7b-instruct', 'mistral-7B-instruct-v0.3', 'glm-4-9b-chat']
        self.retriever = retriever
        self.refiner = self.config['refiner']
        self.model2path = {
            'qwen2-7b-instruct': '/your_path/models/Qwen/Qwen2-7B-Instruct',
            'glm-4-9b-chat': "/your_path/models/ZhipuAI/glm-4-9b-chat",
            'mistral-7B-instruct-v0.3': '/your_path/models/LLM-Research/Mistral-7B-Instruct-v0___3',
            'llama2-7B-chat': '/your_path/models/Llama-2-7b-chat-hf',
            'llama3-8B-instruct': '/your_path/models/LLM-Research/Meta-Llama-3-8B-Instruct',
            'qwen2.5-7b-instruct': '/your_path/Qwen2.5-7B-Instruct',
            'llama3.1-8B-instruct': '/your_path/Llama-3.1-8B-Instruct',
            'llama3.2-1B-instruct': '/your_path/LLM-Research/Llama-3___2-1B-Instruct',
            'llama3.2-3B-instruct': '/your_path/LLM-Research/Llama-3___2-3B-Instruct',
            'qwen2.5-1.5b-instruct': '/your_path/Qwen/Qwen2___5-1___5B-Instruct',
            'qwen2.5-0.5b-instruct': '/your_path/Qwen/Qwen2___5-0___5B-Instruct',
            
            'deepseek-llm-7b-chat': '/your_path/deepseek-ai/deepseek-llm-7b-chat',
            'baichuan2-7b-chat': '/your_path/baichuan-inc/Baichuan2-7B-Chat',
            'qwen2.5-3b-instruct': '/your_path/Qwen/Qwen2___5-3B-Instruct',
            'openelm-3b-instruct': '/your_path/LLM-Research/OpenELM-3B-Instruct',
            'qwen2.5-3b': '/your_path/Qwen/Qwen2___5-3B',
            'gemma-7b-it': '/your_path/LLM-Research/gemma-7b-it',
            'gemma-2-2b-it': '/your_path/LLM-Research/gemma-2-2b-it',
            'gemma-2b-it': '/your_path/AI-ModelScope/gemma-2b-it',
            'qwen2.5-14b-instruct': '/your_path/Qwen/Qwen2___5-14B-Instruct',
            'qwen2.5-32b-instruct': '/your_path/Qwen/Qwen2___5-32B-Instruct',
            'qwen2.5-72B-Instruct': '/your_path/qwen2.5-72B-Instruct',
        }
        self.run_model = {
            'model_name': self.config['model_name'],#'llama3-8B-instruct',#'qwen2-7B-instruct',
            'model_path': self.model2path[self.config['model_name']],
            'max_input_len': self.config['ensembler']['max_input_len'],
            "framework": "hf",
            'type': torch.bfloat16,
            'gpu_use': self.config['ensembler']['gpu_use'],
            'generator_params':
                {
                    'max_tokens': self.config['ensembler']['generator_params']['max_tokens'],
                    'temperature': self.config['ensembler']['generator_params']['temperature'],
                    'top_p': self.config['ensembler']['generator_params']['top_p'],
                    'n': 1,
                    'repetition_penalty': self.config['ensembler']['generator_params']['repetition_penalty'],
                },
            'gpu_num': self.config['gpu_num']
        }
        
    def load_model(self, config, lora=False):
        if config['model_name'] != 'e5':
            if lora:
                if 'falcon' in config['model_name']:
                    model = LLM(
                        config['model_path'],
                        dtype=config['type'],
                        enforce_eager=True,
                        trust_remote_code=True,
                        max_model_len=config['max_input_len'],
                        gpu_memory_utilization=config['gpu_use'],
                        enable_lora=True,
                        max_lora_rank=8,
                        config_class=FalconConfig,
                        tensor_parallel_size=config['gpu_num'],
                    )
                else:
                    model = LLM(
                        config['model_path'],
                        dtype=config['type'],
                        enforce_eager=True,
                        trust_remote_code=True,
                        max_model_len=config['max_input_len'],
                        gpu_memory_utilization=config['gpu_use'],
                        enable_lora=True,
                        max_lora_rank=8,
                        tensor_parallel_size=config['gpu_num'],
                    )
            else:
                if 'falcon' in config['model_name']:
                    model = LLM(
                        config['model_path'],
                        dtype=config['type'],
                        enforce_eager=True,
                        trust_remote_code=True,
                        max_model_len=config['max_input_len'],
                        gpu_memory_utilization=config['gpu_use'],
                        config_class=FalconConfig,
                        tensor_parallel_size=config['gpu_num'],
                    )
                else:
                    model = LLM(
                        config['model_path'],
                        dtype=config['type'],
                        enforce_eager=True,
                        trust_remote_code=True,
                        max_model_len=config['max_input_len'],
                        gpu_memory_utilization=config['gpu_use'],
                        tensor_parallel_size=config['gpu_num'],
                    )
            tokenizer = AutoTokenizer.from_pretrained(config['model_path'], trust_remote_code=True)
            generation_params = dict()
            generation_params.update(config['generator_params'])
            if 'llama' in config['model_name'].lower():
                generation_params['stop_token_ids'] = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            sample_params = SamplingParams(
                **generation_params,
            )
            return model, tokenizer, sample_params
        else:
            model = AutoModel.from_pretrained(config['model_path'], trust_remote_code=True)
            # model = AutoModelForCausalLM.from_pretrained(config['model_path'], trust_remote_code=True, is_decoder=True)
            tokenizer = AutoTokenizer.from_pretrained(config['model_path'], trust_remote_code=True)
            return model, tokenizer, None
    
    def del_model(self, model, tokenizer):
        del tokenizer
        destroy_model_parallel()
        del model.llm_engine.model_executor.driver_worker
        del model
        gc.collect()
        torch.cuda.empty_cache()
    
    def run(self, questions, golden_answers, lora):
        from inference import Generator, Ensembler
        start_time = time.time()
        initial_results = [[] for _ in range(len(questions))]
        results = []
        retrieval_results = [[] for _ in range(len(questions))]
        if self.config['refiner']:
            refine_retrieval_results = [[] for _ in range(len(questions))]
        if 'llama' in self.config['model_name'].lower():
            if 'llama2-7B-chat' in self.model_names:
                self.model_names.remove('llama2-7B-chat')
        elif 'qwen' in self.config['model_name'].lower():
            if 'qwen2-7b-instruct' in self.model_names:
                self.model_names.remove('qwen2-7b-instruct')
        if len(self.model_names) > 3 and self.config['exp_type'] != 'model_generalization':
            self.model_names = random.sample(self.model_names, 3)
        for model_name in self.model_names:
            if self.config['metares_type'] == 'lora':
                naive_path = f'/your_path/results/{self.config["dataset_name"]}/naiveRAG_{model_name}_diffref_lora.jsonl'#，
            else:
                naive_path = f'/your_path/results/{self.config["dataset_name"]}/naiveRAG_{model_name}_diffref.jsonl'
                if self.config['exp_type'] == 'd_a':
                    naive_path = f'/your_path/results/{self.config["dataset_name"]}/naiveRAG_{model_name}_diffref_d_a.jsonl'
            with open(naive_path, 'r', encoding='utf-8') as fr:
                temp_retrieval_results = []
                temp_initial_results = []
                if self.config['refiner']:
                    temp_refine_retrieval_results = []
                    for line in fr:
                        data = ujson.loads(line)
                        temp_retrieval_results.append(data['retrieval result'])
                        temp_initial_results.append(data['answer'])
                        temp_refine_retrieval_results.append(data['refine retrieval result'])
                    for i in range(len(questions)):
                        retrieval_results[i].append(temp_retrieval_results[i])
                        initial_results[i].append(temp_initial_results[i])
                        refine_retrieval_results[i].append(temp_refine_retrieval_results[i])
                else:
                    for line in fr:
                        data = ujson.loads(line)
                        temp_retrieval_results.append(data['retrieval result'])
                        temp_initial_results.append(data['answer'])
                    for i in range(len(questions)):
                        retrieval_results[i].append(temp_retrieval_results[i])
                        initial_results[i].append(temp_initial_results[i])
        if self.config['refiner']:
            retrieval_results = refine_retrieval_results        
        
        model, tokenizer, sample_params = self.load_model(self.run_model, lora)
        if self.config['dataset_name'] == 'fever':
            self.config['system_prompt'] = 3
        elif self.config['dataset_name'] == 'eli5' or self.config['dataset_name'] == 'wikiasp':
            self.config['system_prompt'] = 4
        if self.config['exp_type'] == 'cot':
            self.config['system_prompt'] = 5
        ensembler = Ensembler(model=model, tokenizer=tokenizer, sample_param=sample_params, model_name=self.config['model_name'], config=self.config, prompt_num=self.config['system_prompt'])
        ensembler_results = ensembler.inference(
            questions, initial_results, retrieval_results, self.config['refiner'], rag_type=self.rag_type, lora=lora, lora_path=self.config['lora_path'], prompt_num=self.config['system_prompt']
            )
        results = ensembler_results
        
        if lora:
            self.result_path = '/your_path/results/' + self.config['dataset_name'] + '/' + self.config['RAG_type'] + 'RAG_' + self.config['model_name'] + '_lora.jsonl'
            self.result_path_indent4 = '/your_path/results/' + self.config['dataset_name'] + '/' + self.config['RAG_type'] + 'RAG_' + self.config['model_name'] + '_lora_indent4.jsonl'
        else:
            self.result_path = '/your_path/results/' + self.config['dataset_name'] + '/' + self.config['RAG_type'] + 'RAG_' + self.config['model_name'] + '.jsonl'
            self.result_path_indent4 = '/your_path/results/' + self.config['dataset_name'] + '/' + self.config['RAG_type'] + 'RAG_' + self.config['model_name'] + '_indent4.jsonl'
        if self.config['exp_type'] == 'cot':
            self.result_path = self.result_path.replace(".jsonl", "_cot.jsonl")
            self.result_path_indent4 = self.result_path_indent4.replace(".jsonl", "_cot.jsonl")
        with open(self.result_path, 'w', encoding='utf-8') as fw:
            for question, retrieval_result, meta_result, result, golden_answer in zip(questions, retrieval_results, initial_results, results, golden_answers):#
                temp_dict = dict()
                temp_dict['question'] = question
                temp_dict['retrieval result'] = retrieval_result
                temp_dict_ = {}
                for model_name, meta_res in zip(self.model_names, meta_result):
                    temp_dict_[model_name] = meta_res
                temp_dict['meta results'] = temp_dict_
                temp_dict['answer'] = result
                temp_dict['golden answer'] = golden_answer
                fw.write(ujson.dumps(temp_dict) + '\n')
                
        with open(self.result_path, 'r', encoding='utf-8') as fr, open(self.result_path_indent4, 'w', encoding='utf-8') as fw:
            for line in fr:
                fw.write(ujson.dumps(ujson.loads(line), indent=4) + '\n')
        end_time = time.time()
        print(f'Program run time for {self.rag_type}: {(end_time - start_time)/60}min')
        
    def naive_run(self, questions, golden_answers, lora):
        from inference import Generator
        start_time = time.time()
        results = []
        retrieval_results = []
        if not self.config['exp_type'] == 'd_a':
            retrieval_results_path = f'/your_path/results/{self.config["dataset_name"]}/retrieval_results.jsonl'
            with open(retrieval_results_path, 'r', encoding='utf-8') as fr:
                for line in fr:
                    data = ujson.loads(line)
                    retrieval_results_item = data['retrieval_results']
                    temp_retrieval_results = []
                    temp_retrieval_results.append(retrieval_results_item[0])
                    temp_retrieval_results.extend(random.sample(retrieval_results_item[1:], 4))
                    retrieval_results.append(temp_retrieval_results)
        else:
            retrieval_results_path = f'/your_path/results/{self.config["dataset_name"]}/naiveRAG_llama2-7B-chat_diffref.jsonl'
            with open(retrieval_results_path, 'r', encoding='utf-8') as fr:
                for line in fr:
                    data = ujson.loads(line)
                    retrieval_results.append(data['retrieval result'])
                    
        model, tokenizer, sample_params = self.load_model(self.run_model, lora)
        
        if self.config['dataset_name'] == 'fever':
            generator = Generator(model, tokenizer, sample_params, self.config['model_name'], self.config, num=1)
        elif self.config['dataset_name'] == 'eli5':
            generator = Generator(model, tokenizer, sample_params, self.config['model_name'], self.config, num=2)
        elif self.config['model_name'] == 'wikiasp':
            generator = Generator(model, tokenizer, sample_params, self.config['model_name'], self.config, num=2)
        else:
            generator = Generator(model, tokenizer, sample_params, self.config['model_name'], self.config, num=0)
        results = generator.inference(questions, retrieval_results, rag_type=self.rag_type, lora=lora, lora_path=self.config['lora_path'])
        for i in range(len(results)):
            results[i] = results[i].lstrip('\n')
            results[i] = results[i].rstrip("\n")
            results[i] = results[i].strip(string.punctuation)
        self.del_model(model, tokenizer)
        refine_retrieval_results = results
        if self.config['refiner']:
            from inference import ExtractiveRefiner
            refiner = {
                'model_name': 'e5',#'qwen2.5-7b-instruct',#'llama2-7B-chat',#
                'model_path': self.config['retrieval_model_path'],
                'max_input_len': 4096,
                "framework": "hf",
                'type': torch.bfloat16,
                'stop_token_ids': [151329,
                            151336,
                            151338],
                'gpu_use':self.config['ensembler']['gpu_use'],
                'generator_params':
                    {
                        'max_tokens': 512,
                        'temperature': 0,
                        'top_p': 0.7,
                    },
                'port': '1557'
            }
            model, tokenizer, sample_params = self.load_model(refiner, False)
            refiner = ExtractiveRefiner(model=model, tokenizer=tokenizer, sample_param=sample_params, config=self.config)
            refine_retrieval_results = refiner.batch_run(questions, retrieval_results, results)
        if lora:
            self.config['result_path'] = '/your_path/results/' + self.config['dataset_name'] + '/' + self.config['RAG_type'] + 'RAG_' + self.config['model_name'] + '_diffref_lora.jsonl'
            self.config['result_path_indent4'] = '/your_path/results/' + self.config['dataset_name'] + '/' + self.config['RAG_type'] + 'RAG_' + self.config['model_name'] + '_diffref_lora_indent4.jsonl'
        else:
            self.config['result_path'] = '/your_path/results/' + self.config['dataset_name'] + '/' + self.config['RAG_type'] + 'RAG_' + self.config['model_name'] + '_diffref.jsonl'
            self.config['result_path_indent4'] = '/your_path/results/' + self.config['dataset_name'] + '/' + self.config['RAG_type'] + 'RAG_' + self.config['model_name'] + '_diffref_indent4.jsonl'
        if self.config['exp_type'] == 'd_a':
            self.config['result_path'] = '/your_path/results/' + self.config['dataset_name'] + '/' + self.config['RAG_type'] + 'RAG_' + self.config['model_name'] + '_diffref_d_a.jsonl'
            self.config['result_path_indent4'] = '/your_path/results/' + self.config['dataset_name'] + '/' + self.config['RAG_type'] + 'RAG_' + self.config['model_name'] + '_diffref_d_a_indent4.jsonl'
        with open(self.config['result_path'], 'w', encoding='utf-8') as fw:
            for question, retrieval_result, result, golden_answer, refine_retrieval_result in zip(questions, retrieval_results, results, golden_answers, refine_retrieval_results):
                temp_dict = dict()
                temp_dict['question'] = question
                temp_dict['retrieval result'] = retrieval_result
                temp_dict['refine retrieval result'] = refine_retrieval_result
                temp_dict['answer'] = result
                temp_dict['golden answer'] = golden_answer
                fw.write(ujson.dumps(temp_dict) + '\n')
                
        with open(self.config['result_path'], 'r', encoding='utf-8') as fr, open(self.config['result_path_indent4'], 'w', encoding='utf-8') as fw:
            for line in fr:
                fw.write(ujson.dumps(ujson.loads(line), indent=4) + '\n')
        end_time = time.time()
        print(f'Program run time for {self.rag_type}: {(end_time - start_time)/60}min')
        
    def save_retrieval_results(self, questions, retrieval_results=None):
        import random
        config = copy.deepcopy(self.config)
        if not self.config['refiner']:
            save_path = f'/your_path/results/{self.config["dataset_name"]}/retrieval_results_top5.jsonl'
            model_nemas = [
                'llama2-7B-chat', 'mistral-7B-instruct-v0.3', 'glm-4-9b-chat', 'qwen2-7b-instruct'
            ]
            with open(save_path, 'w', encoding='utf-8') as fw:
                for question, retrieval_result in zip(questions, retrieval_results):
                    temp = {}
                    temp['question'] = question
                    temp['retrieval result'] = retrieval_result
                    fw.write(ujson.dumps(temp) + '\n')
        else:
            save_path = f'/your_path/results/{self.config["dataset_name"]}/retrieval_results_refiner.jsonl'
            refiner = {
                'model_name': 'e5',#'qwen2.5-7b-instruct',#'llama2-7B-chat',#
                'model_path': config['retrieval_model_path'],
                'max_input_len': 512,
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
            model, tokenizer, sample_params = self.load_model(refiner, False)
            from inference import ExtractiveRefiner
            refiner = ExtractiveRefiner(config, model, tokenizer, sample_params)
            results = refiner.batch_run(questions, retrieval_results)
            del tokenizer
            destroy_model_parallel()
            del model
            gc.collect()
            torch.cuda.empty_cache()
            with open(save_path, 'w', encoding='utf-8') as fw:
                for result in results:
                    fw.write(ujson.dumps(result) + '\n')
            
            
    def self_consistency(self, questions, golden_answers, num=3, lora=False):
        from inference import Generator
        from collections import Counter
        start_time = time.time()  
        retrieval_results = []
        retrieval_results_path = f'/your_path/results/{self.config["dataset_name"]}/naiveRAG_{self.config["model_name"]}_diffref.jsonl'
        with open(retrieval_results_path, 'r', encoding='utf-8') as fr:
            for line in fr:
                data = ujson.loads(line)
                retrieval_results_item = data["retrieval result"]
                retrieval_results.append(retrieval_results_item)
        results = [[] for _ in range(len(questions))]
        model, tokenizer, sample_params = self.load_model(self.run_model, lora)
        generator = Generator(model, tokenizer, sample_params, self.config['model_name'], self.config, num=0)
        for i in range(self.config['vote_counts']):
            results_ = generator.inference(questions, retrieval_results, rag_type=self.rag_type, lora=lora, lora_path=self.config['lora_path'])
            # print(results_)
            for j in range(len(results_)):
                results_[j] = results_[j].lstrip('\n')
                results_[j] = results_[j].rstrip("\n")
                results_[j] = results_[j].strip(string.punctuation)
                results[j].append(results_[j])
        # print(results)
                
        final_results = []
        # print(results)
        
        for i, sublist in enumerate(results):
            counts = Counter(sublist)
            max_count = max(counts.values())
            top_elements = [elem for elem, cnt in counts.items() if cnt == max_count]
            chosen = random.choice(top_elements)
            final_results.append(chosen)
            
        self.config['result_path'] = '/your_path/results/' + self.config['dataset_name'] + '/scRAG_' + self.config['model_name'] + '.jsonl'
        self.config['result_path_indent4'] = '/your_path/results/' + self.config['dataset_name'] + '/scRAG_' + self.config['model_name'] + '_indent4.jsonl'
        with open(self.config['result_path'], 'w', encoding='utf-8') as fw:
            for question, retrieval_result, meta_result, result, golden_answer in zip(questions, retrieval_results, results, final_results, golden_answers):
                temp_dict = dict()
                temp_dict['question'] = question
                temp_dict['retrieval result'] = retrieval_result
                temp_dict['meta results'] = meta_result
                temp_dict['answer'] = result
                temp_dict['golden answer'] = golden_answer
                fw.write(ujson.dumps(temp_dict) + '\n')
        with open(self.config['result_path'], 'r', encoding='utf-8') as fr, open(self.config['result_path_indent4'], 'w', encoding='utf-8') as fw:
            for line in fr:
                fw.write(ujson.dumps(ujson.loads(line), indent=4) + '\n')
        
        end_time = time.time()
        print(f'Program run time for {self.rag_type}: {(end_time - start_time)/60}min')    
        
    def cal_logits(self, questions, golden_answers, lora):
        results = []
        system_prompts, user_prompts = None, None
        if 'naive' in self.config['RAG_type']:
            retrieval_results = []
            retrieval_results_path = f'/your_path/results/{self.config["dataset_name"]}/retrieval_results.jsonl'
            with open(retrieval_results_path, 'r', encoding='utf-8') as fr:
                for line in fr:
                    data = ujson.loads(line)
                    retrieval_results_item = data['retrieval_results']
                    temp_retrieval_results = []
                    temp_retrieval_results.append(retrieval_results_item[0])
                    temp_retrieval_results.extend(random.sample(retrieval_results_item[1:], 4))
                    retrieval_results.append(temp_retrieval_results)
            from inference import Generator
            generator = Generator()
            system_prompts, user_prompts = generator.make_prompts(questions, retrieval_results)
        elif 'meta' in self.config['RAG_type']:
            initial_results = [[] for _ in range(len(questions))]
            retrieval_results = [[] for _ in range(len(questions))]
            if 'llama' in self.config['model_name'].lower():
                if 'llama2-7B-chat' in self.model_names:
                    self.model_names.remove('llama2-7B-chat')
            elif 'qwen' in self.config['model_name'].lower():
                if 'qwen2-7b-instruct' in self.model_names:
                    self.model_names.remove('qwen2-7b-instruct')
            for model_name in self.model_names:
                if self.config['metares_type'] == 'lora':
                    naive_path = f'/your_path/results/{self.config["dataset_name"]}/naiveRAG_{model_name}_diffref_lora.jsonl'#，
                else:
                    naive_path = f'/your_path/results/{self.config["dataset_name"]}/naiveRAG_{model_name}_diffref.jsonl'
                with open(naive_path, 'r', encoding='utf-8') as fr:
                    temp_retrieval_results = []
                    temp_initial_results = []
                    for line in fr:
                        data = ujson.loads(line)
                        temp_retrieval_results.append(data['retrieval result'])
                        temp_initial_results.append(data['answer'])
                    for i in range(len(questions)):
                        retrieval_results[i].append(temp_retrieval_results[i])
                        initial_results[i].append(temp_initial_results[i])
            from inference import Ensembler
            ensembler = Ensembler()
            system_prompts, user_prompts = ensembler.make_prompts(questions, initial_results, retrieval_results, self.config['refiner'])
        inputs = [
                    [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_prompt}
                    ] for system_prompt, user_prompt in zip(system_prompts, user_prompts)
                ]
        if 'mistral' in self.config['model_name'].lower() or 'gemma' in self.config['model_name'].lower():
            inputs = [
                [
                    {
                        "role": "user",
                        "content": f'{text[0]["content"]}\n{text[1]["content"]}'
                    }
                ] for text in inputs
            ]
        elif 'baichuan' in self.config['model_name'].lower():
            inputs = [
                f'<reserved_106>{text[0]["content"]}\n{text[1]["content"]}\n<reserved_107>'
                for text in inputs
            ]
        params = {
            'temperature':0,
            'top_p': 0.8,
            'stop': "<|eot_id|>",
            'max_tokens': 128,
            'frequency_penalty':1,
            'logprobs': True,
        }
        from openai import OpenAI
        import math
        client = OpenAI(api_key='none', base_url=f"http://localhost:{self.config['port']}/v1")
        
        results = []
        for i in tqdm(range(len(inputs)), desc='Generating answers w/ probability'):
            completion = client.chat.completions.create(
                model=self.model2path[self.config['model_name']],
                messages=inputs[i],
                **params,
            )
            choice = completion.choices[0]
            output = choice.message.content
            total_logprob = 0
            total_entropy = 0 
            for token_logprob in choice.logprobs.content:
                logprob = token_logprob.logprob
                prob = math.exp(logprob) 

                total_logprob += logprob

                total_entropy -= prob * math.log(prob)  # -p * log(p)

            total_probability = math.exp(total_logprob)
            results.append({
                'answer': output,
                'probability': total_probability,
                'entropy': total_entropy
            })
        
        self.config['result_path'] = '/your_path/results/' + self.config['dataset_name'] + '/' + self.config['RAG_type'] + '_' + self.config['model_name'] + 'generate_prob.jsonl'
        self.config['result_path_indent4'] = '/your_path/results/' + self.config['dataset_name'] + '/' + self.config['RAG_type'] + '_' + self.config['model_name'] + 'generate_prob_indent4.jsonl'
        with open(self.config['result_path'], 'w', encoding='utf-8') as fw:
            for question, result in zip(questions, results):
                data = {
                    'question': question,
                    'answer': result['answer'],
                    'probability': result['probability'],
                    'entropy': result['entropy']
                }
                fw.write(ujson.dumps(data) + '\n')
        with open(self.config['result_path'], 'r', encoding='utf-8') as fr, open(self.config['result_path_indent4'], 'w', encoding='utf-8') as fw:
            for line in fr:
                fw.write(ujson.dumps(ujson.loads(line), indent=4) + '\n')

    def cal_logits_golden_answer(self, questions, golden_answers, lora):
        results = []
        system_prompts, user_prompts = None, None
        if 'naive' in self.config['RAG_type']:
            retrieval_results = []
            retrieval_results_path = f'/your_path/results/{self.config["dataset_name"]}/retrieval_results.jsonl'
            with open(retrieval_results_path, 'r', encoding='utf-8') as fr:
                for line in fr:
                    data = ujson.loads(line)
                    retrieval_results_item = data['retrieval_results']
                    temp_retrieval_results = []
                    temp_retrieval_results.append(retrieval_results_item[0])
                    temp_retrieval_results.extend(random.sample(retrieval_results_item[1:], 4))
                    retrieval_results.append(temp_retrieval_results)
            from inference import Generator
            generator = Generator()
            system_prompts, user_prompts = generator.make_prompts(questions, retrieval_results)
        elif 'meta' in self.config['RAG_type']:
            initial_results = [[] for _ in range(len(questions))]
            retrieval_results = [[] for _ in range(len(questions))]
            if 'llama' in self.config['model_name'].lower():
                if 'llama2-7B-chat' in self.model_names:
                    self.model_names.remove('llama2-7B-chat')
            elif 'qwen' in self.config['model_name'].lower():
                if 'qwen2-7b-instruct' in self.model_names:
                    self.model_names.remove('qwen2-7b-instruct')
            for model_name in self.model_names:
                if self.config['metares_type'] == 'lora':
                    naive_path = f'/your_path/results/{self.config["dataset_name"]}/naiveRAG_{model_name}_diffref_lora.jsonl'#，
                else:
                    naive_path = f'/your_path/results/{self.config["dataset_name"]}/naiveRAG_{model_name}_diffref.jsonl'
                with open(naive_path, 'r', encoding='utf-8') as fr:
                    temp_retrieval_results = []
                    temp_initial_results = []
                    for line in fr:
                        data = ujson.loads(line)
                        temp_retrieval_results.append(data['retrieval result'])
                        temp_initial_results.append(data['answer'])
                    for i in range(len(questions)):
                        retrieval_results[i].append(temp_retrieval_results[i])
                        initial_results[i].append(temp_initial_results[i])
            from inference import Ensembler
            ensembler = Ensembler()
            system_prompts, user_prompts = ensembler.make_prompts(questions, initial_results, retrieval_results, self.config['refiner'])
        inputs = [
                    [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_prompt}
                    ] for system_prompt, user_prompt in zip(system_prompts, user_prompts)
                ]
        if 'mistral' in self.config['model_name'].lower() or 'gemma' in self.config['model_name'].lower():
            inputs = [
                [
                    {
                        "role": "user",
                        "content": f'{text[0]["content"]}\n{text[1]["content"]}'
                    }
                ] for text in inputs
            ]
        elif 'baichuan' in self.config['model_name'].lower():
            inputs = [
                f'<reserved_106>{text[0]["content"]}\n{text[1]["content"]}\n<reserved_107>'
                for text in inputs
            ]
        
        tokenizer = AutoTokenizer.from_pretrained(self.model2path[self.config['model_name']], trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(self.model2path[self.config['model_name']], torch_dtype=torch.bfloat16).cuda()
        if 'mistral' not in self.config['model_name'].lower() and 'baichuan' not in self.config['model_name'].lower() and 'gemma' not in self.config['model_name'].lower():
            inputs = [
                    tokenizer.apply_chat_template(
                    text, tokenize=False, add_generation_prompt=True
                ) for text in inputs
            ]
        elif 'mistral' in self.config['model_name'].lower():
            inputs = [
                    tokenizer.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": f'{text[0]["content"]}\n{text[1]["content"]}'
                        }
                    ], tokenize=False, add_generation_prompt=True, add_model_prefix=True
                ) for text in inputs
            ]
        elif 'baichuan' in self.config['model_name'].lower():
            inputs = [
                f'<reserved_106>{text[0]["content"]}\n{text[1]["content"]}\n<reserved_107>'
                for text in inputs
            ]
        elif 'gemma' in self.config['model_name'].lower():
            
            inputs = [
                    tokenizer.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": f'{text[0]["content"]}\n{text[1]["content"]}'
                        }
                    ], tokenize=False, add_generation_prompt=True, add_model_prefix=True
                ) for text in inputs
            ]
        for i in tqdm(range(len(inputs)), desc='Generating answers w/ probability'):
            input_text = inputs[i]
            target_text = random.choice(golden_answers[i])
            text_prob, text_surprisal = self.cal_prob(target_text, input_text, model, tokenizer)
            results.append([text_prob, text_surprisal])
        self.config['result_path'] = '/your_path/results/' + self.config['dataset_name'] + '/' + self.config['RAG_type'] + '_' + self.config['model_name'] + 'golden_answer_prob.jsonl'
        self.config['result_path_indent4'] = '/your_path/results/' + self.config['dataset_name'] + '/' + self.config['RAG_type'] + '_' + self.config['model_name'] + 'golden_answer_prob_indent4.jsonl'
        with open(self.config['result_path'], 'w', encoding='utf-8') as fw:
            for question, result, golden_answer in zip(questions, results, golden_answers):
                data = {
                    'question': question,
                    'probability': result[0],
                    'surprisal': result[1],
                    'golden_answer': golden_answer
                }
                fw.write(ujson.dumps(data) + '\n')
        with open(self.config['result_path'], 'r', encoding='utf-8') as fr, open(self.config['result_path_indent4'], 'w', encoding='utf-8') as fw:
            for line in fr:
                fw.write(ujson.dumps(ujson.loads(line), indent=4) + '\n')
        
    
    def cal_prob(self, target_text, input_text, model, tokenizer):
        if not isinstance(target_text, str):
            raise ValueError(f"target_text must be a string, got {type(target_text)}")
        model.eval()
        encodings = tokenizer(input_text, return_tensors="pt")
        input_ids = encodings["input_ids"].to(model.device)
        attention_mask = encodings["attention_mask"].to(model.device)
        
        target_encodings = tokenizer(target_text, return_tensors="pt", add_special_tokens=False)
        target_ids = target_encodings["input_ids"][0]  

        total_log_prob = 0.0 
        total_surprisal = 0.0  
        current_input_ids = input_ids
        current_attention_mask = attention_mask
        
        with torch.no_grad():
            for target_id in target_ids:
                outputs = model(input_ids=current_input_ids, attention_mask=current_attention_mask)
                logits = outputs.logits 
                last_logits = logits[:, -1, :]  # [batch_size, vocab_size]
                probs = F.softmax(last_logits, dim=-1)  # [1, vocab_size]
                
                target_prob = probs[0, target_id].item()
                if target_prob == 0:
                    return 0.0, total_surprisal
                
                total_log_prob += torch.log(torch.tensor(target_prob)).item()
                
                surprisal = -torch.log(torch.tensor(target_prob) + 1e-20).item()
                total_surprisal += surprisal
                target_id_tensor = target_id.unsqueeze(0).unsqueeze(0)  # [1, 1]
                current_input_ids = torch.cat([current_input_ids, target_id_tensor.to(model.device)], dim=1)
                current_attention_mask = torch.cat([current_attention_mask, torch.ones_like(target_id_tensor).to(model.device)], dim=1)
        
        total_prob = torch.exp(torch.tensor(total_log_prob)).item()
        return total_prob, total_surprisal
    
    def worst2best(self, questions, golden_answers, lora):
        from inference import Ensembler
        model_names_all = ['gemma-7b-it',
                       'deepseek-llm-7b-chat',
                       'llama2-7B-chat',  
                       'baichuan2-7b-chat',
                       'mistral-7B-instruct-v0.3',
                       'glm-4-9b-chat',
                        'qwen2-7b-instruct',]
        model_num = list(range(1, len(model_names_all)+1))
        if self.config['model_num'] != 0:
            model_num = [self.config['model_num']]
                        
        model, tokenizer, sample_params = self.load_model(self.run_model, lora)
        
        
        for i in model_num:
            model_names = model_names_all[:i]
            initial_results = [[] for _ in range(len(questions))]
            results = []
            retrieval_results = [[] for _ in range(len(questions))]
            for model_name in model_names:
                naive_path = f'/your_path/results/{self.config["dataset_name"]}/naiveRAG_{model_name}_diffref.jsonl'
                with open(naive_path, 'r', encoding='utf-8') as fr:
                    temp_retrieval_results = []
                    temp_initial_results = []
                    for line in fr:
                        data = ujson.loads(line)
                        temp_retrieval_results.append(data['retrieval result'])
                        temp_initial_results.append(data['answer'])
                    for i in range(len(questions)):
                        retrieval_results[i].append(temp_retrieval_results[i])
                        initial_results[i].append(temp_initial_results[i])
            
            ensembler = Ensembler(model=model, tokenizer=tokenizer, sample_param=sample_params, model_name=self.config['model_name'], config=self.config, prompt_num=self.config['system_prompt'])
            results = ensembler.inference(
                questions, initial_results, retrieval_results, self.config['refiner'], rag_type=self.rag_type, lora=lora, lora_path=self.config['lora_path'], prompt_num=self.config['system_prompt']
                )
            results = [result.strip() for result in results]
            
            from evaluator.evaluator import Evaluator
            import datetime, time
            
            
            config = {
                'metrics': ['em', 'f1', 'acc'],
                'dataset_name': self.config['dataset_name'],
                'acc_in': False
            }
            evaluator = Evaluator(config)
            eval_results = evaluator.evaluate(results, golden_answers)
            eval_results = {k: round(v, 4) for k, v in eval_results.items()}
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{current_time}] Result: {eval_results}, Generators: {self.config['model_name']}, RAG type: {self.rag_type}, Additional: model from worst to best for {len(model_names)} models\n"
            result_log = '/your_path/results/' + self.config['dataset_name'] + '/result.log'
            log_file = open(result_log, 'a')
            log_file.write(log_entry)
            print(eval_results)
        
    def best2worst(self, questions, golden_answers, lora):
        from inference import Ensembler
        model_names_all = ['qwen2-7b-instruct',
                       'glm-4-9b-chat',
                       'mistral-7B-instruct-v0.3',
                       'baichuan2-7b-chat',
                       'llama2-7B-chat', 
                       'deepseek-llm-7b-chat', 
                       'gemma-7b-it', ]
        model_num = list(range(1, len(model_names_all)+1))
        rng = np.random.default_rng(1557)
        if self.config['model_num'] != 0:
            model_num = [self.config['model_num']]
            model_names_all = [
                'qwen2-7b-instruct',
                'glm-4-9b-chat',
                'mistral-7B-instruct-v0.3',
                'baichuan2-7b-chat',
                'gemma-7b-it',
            ]
                        
        model, tokenizer, sample_params = self.load_model(self.run_model, lora)
        
        
        for i in model_num:
            model_names = model_names_all[:i]
            initial_results = [[] for _ in range(len(questions))]
            results = []
            retrieval_results = [[] for _ in range(len(questions))]
            for model_name in model_names:
                naive_path = f'/your_path/results/{self.config["dataset_name"]}/naiveRAG_{model_name}_diffref.jsonl'
                with open(naive_path, 'r', encoding='utf-8') as fr:
                    temp_retrieval_results = []
                    temp_initial_results = []
                    for line in fr:
                        data = ujson.loads(line)
                        temp_retrieval_results.append(data['retrieval result'])
                        temp_initial_results.append(data['answer'])
                    for i in range(len(questions)):
                        retrieval_results[i].append(temp_retrieval_results[i])
                        initial_results[i].append(temp_initial_results[i])
            
            ensembler = Ensembler(model=model, tokenizer=tokenizer, sample_param=sample_params, model_name=self.config['model_name'], config=self.config, prompt_num=self.config['system_prompt'])
            results = ensembler.inference(
                questions, initial_results, retrieval_results, self.config['refiner'], rag_type=self.rag_type, lora=lora, lora_path=self.config['lora_path'], prompt_num=self.config['system_prompt']
                )
            results = [result.strip() for result in results]
            
            from evaluator.evaluator import Evaluator
            import datetime, time
            
            
            config = {
                'metrics': ['em', 'f1', 'acc'],
                'dataset_name': self.config['dataset_name'],
                'acc_in': False
            }
            evaluator = Evaluator(config)
            eval_results = evaluator.evaluate(results, golden_answers)
            eval_results = {k: round(v, 4) for k, v in eval_results.items()}
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{current_time}] Result: {eval_results}, Generators: {self.config['model_name']}, RAG type: {self.rag_type}, Additional: model from best to worst for {len(model_names)} models\n"
            result_log = '/your_path/results/' + self.config['dataset_name'] + '/result.log'
            log_file = open(result_log, 'a')
            log_file.write(log_entry)
            print(eval_results)

        
    def SAE(self, questions, golden_answers, lora):
        from inference import Generator, Ensembler
        start_time = time.time()
        initial_results = []
        results = []
        retrieval_results = []
        
        SAE_path = f'/your_path/results/{self.config["dataset_name"]}/scRAG_{self.config["model_name"]}.jsonl'
        with open(SAE_path, 'r', encoding='utf-8') as fr:
            for line in fr:
                data = ujson.loads(line)
                initial_results.append(data['meta results'])
                retrieval_results.append([data['retrieval result']] * len(data['meta results']))
        
        model, tokenizer, sample_params = self.load_model(self.run_model, lora)
        if self.config['exp_type'] == 'cot':
            self.config['system_prompt'] = 5
        
        ensembler = Ensembler(model=model, tokenizer=tokenizer, sample_param=sample_params, model_name=self.config['model_name'], config=self.config, prompt_num=self.config['system_prompt'])
        results = ensembler.inference(
            questions, initial_results, retrieval_results, self.config['refiner'], rag_type=self.rag_type, lora=lora, lora_path=self.config['lora_path'], prompt_num=self.config['system_prompt']
            )
        self.config['result_path'] = '/your_path/results/' + self.config['dataset_name'] + '/SAE_' + self.config['model_name'] + '.jsonl'
        self.config['result_path_indent4'] = '/your_path/results/' + self.config['dataset_name'] + '/SAE_' + self.config['model_name'] + '_indent4.jsonl'
        with open(self.config['result_path'], 'w', encoding='utf-8') as fw:
            for question, retrieval_result, meta_result, result, golden_answer in zip(questions, retrieval_results, initial_results, results, golden_answers):#
                temp_dict = dict()
                temp_dict['question'] = question
                temp_dict['retrieval result'] = retrieval_result
                temp_dict['meta results'] = meta_result
                temp_dict['answer'] = result
                temp_dict['golden answer'] = golden_answer
                fw.write(ujson.dumps(temp_dict) + '\n')
        with open(self.config['result_path'], 'r', encoding='utf-8') as fr, open(self.config['result_path_indent4'], 'w', encoding='utf-8') as fw:
            for line in fr:
                fw.write(ujson.dumps(ujson.loads(line), indent=4) + '\n')
        
        end_time = time.time()
        print(f'Program run time for {self.rag_type}: {(end_time - start_time)/60}min')
