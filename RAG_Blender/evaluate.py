from evaluator.evaluator import Evaluator
import datetime, time
import copy
from tqdm import tqdm
import ujson
import torch
import os
from utils import get_dataset
import datetime, time
from run import config as config_
import argparse
# from multi_llms_for_CoT.scripts.run_multiRAG import config_inference
import re
import ast
import json
# from inference import Checker, Generator, Modifier, Rethinker
# from inference_choose import Generator, Classifier, Modifier, Simplifier

config = {
    'metrics': ['em', 'f1', 'acc'],#, 'bleu'
    'save_dir': '/your_path/multi_llms_for_CoT/datasets/hotpotqa/LLM_ensemble_results.jsonl',
    'dataset_name': 'nq',
    'rag_naive': '/your_path/multi_llms_for_CoT/results/2wikimultihopqa/naiveRAG.jsonl',
    'RAG_type': 'check',
    'refiner_name': None,#'llama3-8B-instruct',
    'metric_setting':
        {
            'retrieval_recall_topk': 5,
            'tokenizer_name': 'llama3',
        }
}
config.update(config_)

def fix_missing_quote(s):
    if isinstance(s, str) and s.strip().endswith("}"):
        stripped_s = s.strip()
        if stripped_s[-2] != "'":
            s = stripped_s[:-1] + "'" + stripped_s[-1]
    return s

def extract_numbers(s):
    numbers = re.findall(r'\b\d+\b', s)
    if numbers:
        return numbers
    else:
        return [s]
    
def extract_first_line(input_str):
    newline_index = input_str.find('\n')
    
    if newline_index != -1:
        return input_str[:newline_index]
    return input_str

all_counts = 0
def extract_answer(s):
    s = s.strip()
    
    if s.startswith('"') and s.endswith('"'):
        s = s[1:-1]
    
    try:
        parsed = json.loads(s)
        if isinstance(parsed, dict) and 'answer' in parsed:
            value = parsed['answer']
            if isinstance(value, str):
                return value
            elif isinstance(value, (int, float)):
                return str(value)
            elif isinstance(value, bool):
                return str(value).lower()
            elif value is None:
                return 'null'
            else:
                return json.dumps(value)
    except json.JSONDecodeError:
        pass
    
    pattern = r'''
        "answer"                     
        \s*:\s*                      
        (?:                          
            "(?P<string>(?:\\.|[^"\\])*)"   
            |                        
            (?P<number>[\d,]+)       
            |                        
            (?P<word>[^\s,}]+)       
            |                        
            (?P<bool_null>true|false|null) 
        )
    '''
    
    match = re.search(pattern, s, re.IGNORECASE | re.VERBOSE)
    if match:
        if match.group('string') is not None:
            answer = match.group('string')
            try:
                answer = bytes(answer, "utf-8").decode("unicode_escape")
            except UnicodeDecodeError:
                pass
            return answer
        elif match.group('number') is not None:
            answer = match.group('number')
        elif match.group('word') is not None:
            answer = match.group('word')
            return answer
        elif match.group('bool_null') is not None:
            answer = match.group('bool_null')
            return answer.lower() if answer.lower() in ['true', 'false', 'null'] else answer
    global all_counts
    all_counts += 1
    return s

def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_type', type=str, default='naive')
    parser.add_argument('--dataset_name', type=str, default='hotpotqa')
    parser.add_argument('--pipeline_type', type=str, default='multi_hlcn')
    parser.add_argument('--acc_in', action='store_true')
    parser.add_argument('--f1_t', type=float, default=0.5)
    parser.add_argument('--model_name', type=str, default='llama3-8B-instruct')
    parser.add_argument('--lora', action='store_true')
    parser.add_argument('--train_times', type=int, default=7500)
    parser.add_argument('--refiner', action='store_true')
    parser.add_argument('--additional', type=str, default='initial')
    parser.add_argument('--epoch', type=float, default=3)
    args = parser.parse_args()
    rag_type = args.eval_type
    result_log = '/your_path/results/' + args.dataset_name + '/result.log'
    if args.dataset_name == 'eli5' or args.dataset_name == 'wikiasp':
        config['metrics'] = ['f1', 'rouge-l']#'bleu', 
    config['RAG_type'] = args.eval_type
    if args.lora:
        if args.eval_type == 'naive':
            config['result_path'] = '/your_path/results/' + args.dataset_name + '/' + args.eval_type + 'RAG_' + args.model_name + '_diffref_lora.jsonl'
        else:
            config['result_path'] = '/your_path/results/' + args.dataset_name + '/' + args.eval_type + 'RAG_' + args.model_name + '_lora.jsonl'#_diffref
    elif args.eval_type == 'naive':
        config['result_path'] = '/your_path/results/' + args.dataset_name + '/' + args.eval_type + 'RAG_' + args.model_name + '_diffref.jsonl'#
    else: 
        config['result_path'] = '/your_path/results/' + args.dataset_name + '/' + args.eval_type + 'RAG_' + args.model_name + '.jsonl'#
    if args.eval_type.lower() == 'one_system':
        config['result_path'] = config['result_path'].replace('.jsonl', '_one_system.jsonl')
    elif args.additional == 'other_rag':
        config['result_path'] = '/your_path/results/other_RAG/' + args.dataset_name + '/metaRAG_' + args.model_name + '.jsonl'
    config['dataset_name'] = args.dataset_name
    config['acc_in'] = args.acc_in
    config['f1_t'] = args.f1_t
    if args.eval_type.lower() == 'sae':
        config['result_path'] = '/your_path/results/' + args.dataset_name + '/SAE_' + args.model_name + '.jsonl'
    elif args.additional == 'cot':
        config['result_path'] = config['result_path'].replace('.jsonl', '_cot.jsonl')
    if 'train_eval' not in args.additional:
        results = get_dataset(config, data_dir='result_path', value='answer')
    else:
        config['result_path'] = f'/your_path/scripts/train/{args.dataset_name}_source_datas.jsonl'
        results = get_dataset(config, data_dir='result_path', value='response')
    results = [result.strip() for result in results]
    answers = []
    if 'cot' in args.additional:
        for i in tqdm(range(len(results))):
            s = results[i]
            s = s.strip()
            answers.append(extract_answer(s))
        results = answers     
            
    if 'train_eval' not in args.additional:
        golden_answers = get_dataset(config, data_dir='result_path', value='golden answer')
    else:
        golden_answers = get_dataset(config, data_dir='result_path', value='golden_answers')

    evaluator = Evaluator(config)
    if 'gemma' in args.model_name:
        results = [extract_first_line(result) for result in results]
    eval_results = evaluator.evaluate(results, golden_answers)
    eval_results = {k: round(v, 4) for k, v in eval_results.items()}
    print(eval_results)
    
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # generators = config['ensembler']['model_name']
    generators = args.model_name
    train_times = args.train_times if args.lora else 0
    log_entry = f"[{current_time}] Result: {eval_results}, Generators: {generators}, RAG type: {rag_type}, Refiner: {args.refiner}, Lora: {args.lora}, Train times: {train_times}, Additional: {args.additional}\n"
    log_file = open(result_log, 'a')
    log_file.write(log_entry)
    
if __name__ == '__main__':
    evaluate()
    