import torch
import re
from typing import List
import numpy as np
from transformers import AutoTokenizer
from vllm.lora.request import LoRARequest
import os
from utils import pooling
from tqdm import tqdm
import ujson
rng = np.random.default_rng(1557)
import random

class BaseInference:
    def __init__(self, model=None, tokenizer=None, sample_param=None, model_name=None):
        self.model = model
        self.tokenizer = tokenizer
        self.sample_param = sample_param
        self.model_name = model_name
        self.system_prompt = ()
        self.system_prompt_reason = ()
        self.user_prompt_reason = ()
        self.user_prompt = ()
        
    def make_prompts(self, *args):
        pass
    
    def make_prompts_reason(self, *args):
        pass
    
    def format_reference(self, retrieval_result):
        format_reference = ""
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item["contents"]
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            text = re.sub(r'[\{\}]', '', text)
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference
    
    def format_example_reference(self, retrieval_result, answer):
        format_reference = ""
        for idx, doc_item in enumerate(retrieval_result):
            format_reference += f"Doc {idx+1}(Title: {answer}) {doc_item}\n"
        return format_reference
    
    def format_answers(self, answers):
        rng.shuffle(answers)
        format_answers = ""
        for idx, answer in enumerate(answers):
            format_answers += f"[{idx+1}] {answer}\n"
        return format_answers
    
    def inference(self, *args, rag_type='meta', api_key='my_key', base_url=None, model_path=None, lora=False, lora_path=None, prompt_num=0, return_inputs=False, exp_type='main'):
        system_prompts, user_prompts = self.make_prompts(*args)
        inputs = [
                    [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ] for system_prompt, user_prompt in zip(system_prompts, user_prompts)
        ]
        if 'mistral' not in self.model_name and 'baichuan' not in self.model_name and 'gemma' not in self.model_name:
            inputs = [
                    self.tokenizer.apply_chat_template(
                    text, tokenize=False, add_generation_prompt=True
                ) for text in inputs
            ]
        elif 'mistral' in self.model_name:
            inputs = [
                    self.tokenizer.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": f'{text[0]["content"]}\n{text[1]["content"]}'
                        }
                    ], tokenize=False, add_generation_prompt=True, add_model_prefix=True
                ) for text in inputs
            ]
        elif 'baichuan' in self.model_name:
            inputs = [
                f'<reserved_106>{text[0]["content"]}\n{text[1]["content"]}\n<reserved_107>'
                for text in inputs
            ]
        elif 'gemma' in self.model_name:
            
            inputs = [
                    self.tokenizer.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": f'{text[0]["content"]}\n{text[1]["content"]}'
                        }
                    ], tokenize=False, add_generation_prompt=True, add_model_prefix=True
                ) for text in inputs
            ]
            
            
        if lora:
            print('Using LoRA')
            outputs = self.model.generate(
                inputs,
                self.sample_param,
                lora_request=LoRARequest("sql-lora", 1, lora_path)
            )
        else:
            outputs = self.model.generate(
                inputs,
                self.sample_param
            )
        if return_inputs:
            return [output_.outputs[0].text for output_ in outputs], inputs
        else:
            if exp_type == 'sc':
                return [[output_.text for output_ in output.outputs] for output in outputs]
            else:
                return [output_.outputs[0].text for output_ in outputs]#, inputs
    
class ExtractiveRefiner(BaseInference):
    def __init__(self, config, model=None, tokenizer=None, sample_param=None):
        super().__init__(model, tokenizer, sample_param)
        self.topk = config["refiner_topk"]
        self.pooling_method = config["refiner_pooling_method"]
        self.encode_max_length = config["refiner_encode_max_length"]
        self.model, self.tokenizer, self.sample_param = model, tokenizer, sample_param
        
    def encode(self, query_list: List[str], is_query=True):
        if is_query:
            query_list = [f"query: {query}" for query in query_list]
        else:
            query_list = [f"passage: {query}" for query in query_list]

        inputs = self.tokenizer(
            query_list, max_length=self.encode_max_length, padding=True, truncation=True, return_tensors="pt"
        )
        inputs = {k: v.cuda() for k, v in inputs.items()}#

        if "T5" in type(self.model).__name__:
            # T5-based retrieval model
            decoder_input_ids = torch.zeros((inputs["input_ids"].shape[0], 1), dtype=torch.long).to(
                inputs["input_ids"].device
            )
            output = self.model(**inputs, decoder_input_ids=decoder_input_ids, return_dict=True)
            query_emb = output.last_hidden_state[:, 0, :]

        else:
            self.model = self.model.to(inputs["input_ids"].device)
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(
                output.pooler_output, output.last_hidden_state, inputs["attention_mask"], self.pooling_method
            )

        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32)
        return query_emb

    def batch_run(self, questions, retrieval_results, answers, batch_size=30):
        retrieval_results = [
            ["\n".join(doc_item["contents"].split("\n")[1:]) for doc_item in item_result]
            for item_result in retrieval_results
        ]

        sent_lists = retrieval_results
        score_lists = []  # matching scores, size == sent_lists
        for idx in tqdm(range(0, len(questions), batch_size), desc="Refining process: "):
            batch_questions = questions[idx : idx + batch_size]
            batch_answers = answers[idx : idx + batch_size]
            batch_questions = [question + "\n" + answer for question, answer in zip(batch_questions, batch_answers)]
            batch_sents = sent_lists[idx : idx + batch_size]

            question_embs = self.encode(batch_questions, is_query=True)
            sent_embs = self.encode(sum(batch_sents, []), is_query=False)  # n*d
            scores = question_embs @ sent_embs.T
            start_idx = 0
            for row_score, single_list in zip(scores, sent_lists):
                row_score = row_score.tolist()
                score_lists.append(row_score[start_idx : start_idx + len(single_list)])
                start_idx += len(single_list)

        # select topk sents
        retain_lists = []
        for sent_scores, sent_list in zip(score_lists, sent_lists):
            if len(sent_scores) < self.topk:
                retain_lists.append(sent_list)
                continue
            topk_idxs = torch.topk(torch.Tensor(sent_scores), self.topk).indices.tolist()
            retain_lists.append([sent_list[idx] for idx in sorted(topk_idxs)])

        return [" ".join(sents) for sents in retain_lists]
    
class Generator(BaseInference):
    def __init__(self, model=None, tokenizer=None, sample_param=None, model_name=None, config=None, num=0):
        self.config = config
        super().__init__(model, tokenizer, sample_param, model_name)
        self.system_prompt = [
            (
                "Answer the question based on the given document. "
                "Only give me the answer and do not output any other words. "
                "\nThe following are given documents.\n\n{reference}"
            ),
            (
                "Determine whether the fact in the question is correct according to the facts in the given documents. "
                "If the question is correct, output 'SUPPORTS', otherwise output 'REFUTES' due to the fact contradiction. "
                "Onlu give me a 'SUPPORTS' or 'REFUTES', and do not output any other words. "
                "\nThe following are given documents.\n\n{reference}"
            ),
            (
                "Write a detailed and coherent response to the following question based on the provided documents. "
                "Your answer should be a well-structured paragraph that fully addresses the question. "
                "Only give me the answer and do not output any other words. "
                "\nThe following are given documents.\n\n{reference}"
            )
        ]
        self.system_prompt = self.system_prompt[num]
        if model_name is not None:
            if 'deepseek' in model_name or 'gemma' in model_name:
                self.system_prompt = (
                    "Answer the question based on the given document. "
                    "Only give me a final answer and do not output any other words. "
                    "\nThe following are given documents.\n\n{reference}"
                )
        self.system_prompt_reason = (
            "You are an advanced system specialized in analyzing information. Below are some external data, a question and an answer provided by a distinct system "
            "based on that external data to the given question. The information is given in JSON format. Your task is to thoroughly analyze the data provided, "
            "and then judge the correctness of the answer provided. Only output your reasons for your judgment and do not output any other words.\n"
            "Here are some examples of how to complete the task:\n"
            "<example 1>\n"
            "{'question': 'What is the largest mammal in the world?', 'external data': 'Doc 1(Title: The Largest Mammal on Earth) The blue whale (Balaenoptera musculus) "
            "is the largest mammal and indeed the largest animal to have ever existed on Earth...\nDoc 2(Title: Why Blue Whales Hold the Record for Size) "
            "Blue whales are unmatched in size, largely due to their unique adaptations to marine life and their food sources...\n...', "
            "'answer': The largest mammal in the world is the blue whale.}\n\nYour answer: The answer is correct. According to the Doc 1, ..., and according to the Doc 2, ..., so the largest mammal in the world is the blue whale.\n\n"
            "<example 2>\n"
            "{'question': 'What is the highest mountain in the world?', 'external data': 'Doc 1(Title: What is the Highest Mountain in the World?) "
            "The highest mountain in the world is Mount Qomolangma, which reaches an impressive height of approximately 8,848 meters (29,029 feet) "
            "above sea level...\nDoc 2(Title: Conquering Earth's Tallest Peak: Mount Qomolangma) This iconic peak, "
            "situated in the Himalayas on the border of Nepal and Tibet, was first successfully summited in 1953 by Sir Edmund Hillary and Tenzing Norgay...\n...', "
            "'answer': The highest mountain in the world is Mount Tai.}\n\nYour answer: The answer is incorrect. "
            "According to the Doc 1, ..., and according to the Doc 2, ..., so the highest mountain in the world is Qomolangma.\n\n"
        )
        self.user_prompt = (
            "Question: {question}"
        )
        self.user_prompt_reason = (
            "{datas}\n"
            "Your answer: "
        )
        
    def make_prompts(self, questions, retrieval_results):
        if self.config == None or 'refiner' not in self.config or not self.config['refiner']:
            format_references = [self.format_reference(retrieval_result) for retrieval_result in retrieval_results]
        else:
            format_references = retrieval_results
        input_params = [{'question': question, 'reference': format_reference} for question, format_reference in zip(questions, format_references)]
        return [self.system_prompt.format(**input_param) for input_param in input_params], [self.user_prompt.format(**input_param) for input_param in input_params]
    
    def make_prompts_reason(self, questions, retrieval_results, answers):
        format_references = [self.format_reference(retrieval_result) for retrieval_result in retrieval_results]
        input_params = []
        for question, format_reference, answer in zip(questions, format_references, answers):
            temp_dict = {}
            temp_dict['question'] = question
            temp_dict['external data'] = format_reference
            temp_dict['answer'] = answer
            input_params.append(temp_dict)
        return [self.system_prompt_reason for _ in input_params], [self.user_prompt_reason.format(datas=input_param) for input_param in input_params]
    
    
class Ensembler(BaseInference):
    def __init__(self, model=None, tokenizer=None, sample_param=None, model_name=None, config=None, prompt_num=0):
        super().__init__(model, tokenizer, sample_param, model_name)
        self.config = config
        self.system_prompts = [
        """Answer the question based on the given external data from {num} systems' information. Each system's information contains reference documents and an answer derived from those documents. Analyze the content of all reference documents. Use the content from all systems to synthesize a single, final answer. Only give me a final answer and do not output any other words. The following are the given external systems' data:\n\n{reference}""",
        """Answer the question based on the given external data from {num} systems' information. Each system's information contains reference documents and an answer derived from those documents. The correctness of each system's data may vary. Analyze the content of all reference documents. Use the content from all systems to synthesize a single, final answer. Only give me the final answer and do not output any other words. The following are the given external systems' data:\n\n{reference}""",
        """Answer the question based on the given external data from {num} systems' information. Each system's information contains reference documents and an answer derived from those documents. The correctness of each system's data may vary, and the ability of each system to answer different types of questions varies. Analyze the content of all reference documents. Use the content from all systems to synthesize a single, final answer. Only give me the final answer and do not output any other words. The following are the given external systems' data:\n\n{reference}""",
        """Determine whether the fact in the question is correct based on the given external data from {num} systems' information. Each system's information contains reference documents and a judgment derived from those documents. Use the content from all systems to synthesize a single, final judgment. If the question is correct, output 'SUPPORTS', otherwise output 'REFUTES' due to the fact contradiction. Onlu give me a 'SUPPORTS' or 'REFUTES', and do not output any other words. The following are the given external systems' data:\n\n{reference}""",
        """Write a detailed and coherent response to the following question based on the given external data from {num} systems' information. Each system's information contains reference documents and a response derived from those documents. Use the content from all systems to synthesize a final response. Only give me the final response and do not output any other words. The following are the given external systems' data:\n\n{reference}""",
        """Answer the question based on the given external data from {num} systems' information. Each system's information contains reference documents and an answer derived from those documents. Here is things to pay attention to:
- First analyze the content of all reference documents.
- Then provide step-by-step reasoning on how to answer the question.
- In the reasoning, if you need to copy paste some sentences from the context, include them in ##begin_quote## and ##end_quote##. This would mean that things outside of ##begin_quote## and ##end_quote## are not directly copy paste from the context.
- End your response with the final answer, and the answer should be succinct, such as a few words, but not a complete sentence.
- You should return your output in JSON format as follows:
  {{\"reason\": <your reason>, \"answer\": <your answer>}}
Strictly follow the required format and content output, and do not output any other words.
        """,
        ]
        self.system_prompt = self.system_prompts[prompt_num].lstrip(' ')
        self.system_prompt = self.system_prompt.rstrip(' ')
        self.prompt_num = prompt_num
        self.user_prompt = (
            "Question: {question}"
        )
        if self.prompt_num == 5:
            self.user_prompt = (
            "Question: {question}\n\n"
            "Systems' information: \n{reference}\n\n"
        )
        
    def make_prompts(self, questions, answers, retrieval_results, refiner):#
        input_params = []
        for question, answer, retrieval_result in zip(questions, answers, retrieval_results):
            temp_dict = {}
            temp_dict['question'] = question
            format_answers = ''
            for idx, (answer_item, retrieval_result_item) in enumerate(zip(answer, retrieval_result)):
                format_answers += f"system {idx+1}:\n\nreference documents:\n"
                format_answers += self.format_reference(retrieval_result_item)
                format_answers += f"\nanswer:\n{answer_item}\n\n"
            temp_dict['reference'] = format_answers
            temp_dict['num'] = len(answer)
            input_params.append(temp_dict)
        
        return [self.system_prompt.format(**input_param) for input_param in input_params], [self.user_prompt.format(**input_param) for input_param in input_params]
 