import linecache
import json
import random
from torch.utils.data import Dataset
import re


class FineTuningQADataset(Dataset):
    def __init__(self, filename, with_ret=False, zero_shot=True, ret_passages=5, model_name=None):
        super(FineTuningQADataset, self).__init__()
        self._filename = filename
        self._with_ret = with_ret
        self._zero_shot = zero_shot
        self._num_ret_passages = ret_passages
        self._model_name = model_name
        with open(filename, "r", encoding="utf-8") as f:
            self._total_data = len(f.readlines())
    
    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 1)
        sample = json.loads(line)
        ret_passages = ""

        ret_passages = sample["meta_results"]
        if self._model_name in sample['meta_models']:
            idx = sample['meta_models'].index(self._model_name)
            ret_passages = ret_passages[idx]['retrieval result']
        elif 'llama' in self._model_name:
            idx = sample['meta_models'].index('llama2-7B-chat')
            ret_passages = ret_passages[idx]['retrieval result']
        elif 'qwen' in self._model_name:
            idx = sample['meta_models'].index('qwen2-7b-instruct')
            ret_passages = ret_passages[idx]['retrieval result']
        else:
            ret_passages = random.choice(ret_passages)
            ret_passages = ret_passages['retrieval result']
        ret_passages = self._gen_ret_results(ret_passages)
        question = sample["question"]
        
        if isinstance(sample["golden_answers"], list):
            target = random.choice(sample["golden_answers"])
        else:
            target = sample["golden_answers"]

        batch = {
            "reference": ret_passages,
            "question": question,
            "completion": target  # completion
        }

        return batch

    def __len__(self):
        return self._total_data

    def _gen_prompt(self, example, include_answer=False):
        prompt = f"Question: {example['question']}\n"  # prompt
        return prompt

    def _gen_ret_results(self, ret_passages):
        prompt = ""
        random.shuffle(ret_passages)
        for idx, p in enumerate(ret_passages):
            content = p["contents"]
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            text = re.sub(r'[\{\}]', '', text)
            prompt += f"Doc {idx+1}(Title: {title}) {text}\n"
        prompt += "\n"
        return prompt
    
class FineTuningMetaRAG(Dataset):
    def __init__(self, filename, with_ret=False, zero_shot=True, ret_passages=5, model_name=None, refiner=False, prompt_num=0, metarag_num=3):
        super(FineTuningMetaRAG, self).__init__()
        self._filename = filename
        self._num_ret_passages = ret_passages
        self._model_name = model_name
        self._refiner = refiner
        self._prompt_num = prompt_num
        self._metarag_num = metarag_num
        with open(filename, "r", encoding="utf-8") as f:
            self._total_data = len(f.readlines())
    
    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 1)
        sample = json.loads(line)
        question = sample["question"]
        answers = [item['answer'] for item in sample["meta_results"]]
        if isinstance(sample["golden_answers"], list):
            target = random.choice(sample["golden_answers"])
        else:
            target = sample["golden_answers"]
        if self._prompt_num == 5:
            target = sample['response']
            target = str(target)
            # target = sample['response']
        if self._metarag_num < len(answers):
            if self._model_name in sample['meta_models']:
                idx = sample['meta_models'].index(self._model_name)
            elif self._model_name == 'llama3.1-8B-instruct' or self._model_name == 'llama3-8B-instruct':
                idx = sample['meta_models'].index('llama2-7B-chat')
            elif self._model_name == 'qwen2.5-7b-instruct':
                idx = sample['meta_models'].index('qwen2-7b-instruct')
            else:
                idx = random.randint(0, len(sample["meta_results"])-1)
            answers.pop(idx)
            if not self._refiner:
                meta_results = sample["meta_results"]
                retrieval_results = [item['retrieval result'] for item in meta_results]
                retrieval_results.pop(idx)
                reference = self._gen_prompt(retrieval_results, answers)
            else:
                assert 'refiner_results' in sample
                meta_results = sample['refiner_results']
                meta_results.pop(idx)
                reference = self._gen_prompt(meta_results, answers)
        else:
            if not self._refiner:
                meta_results = sample["meta_results"]
                retrieval_results = [item['retrieval result'] for item in meta_results]
                reference = self._gen_prompt(retrieval_results, answers)
            else:
                assert 'refiner_results' in sample
                meta_results = sample['refiner_results']
                reference = self._gen_prompt(meta_results, answers)
            
        batch = {
            'num': len(answers),
            'question': question,
            'reference': reference,
            'completion': target
        }

        return batch

    def __len__(self):
        return self._total_data

    def _gen_prompt(self, retrieval_results, answers):#question, meta_results, 
        format_answers = ''
        for idx, (answer, retrieval_result) in enumerate(zip(answers, retrieval_results)):
            format_answers += f"System {idx+1}:\n\nreference documents:\n"
            if not self._refiner:
                for idx_, doc_item in enumerate(retrieval_result):
                    content = doc_item["contents"]
                    title = content.split("\n")[0]
                    text = "\n".join(content.split("\n")[1:])
                    text = re.sub(r'[\{\}]', '', text)
                    format_answers += f"Doc {idx_+1}(Title: {title}) {text}\n"
            else:
                format_answers += retrieval_result
            format_answers += f"\nAnswer:\n{answer}\n\n"
        return format_answers
    
        

