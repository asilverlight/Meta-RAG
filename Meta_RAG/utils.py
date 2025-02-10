import json
import re
import importlib

def get_dataset(config, data_dir='data_dir', value='question'):
    
    data_path = config[data_dir]
    questions = []
    with open(data_path, 'r', encoding='utf-8') as fr:
        for line in fr:
            data = json.loads(line)
            questions.append(data[value])
    return questions

def pooling(pooler_output, last_hidden_state, attention_mask=None, pooling_method="mean"):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")
    
def remove_substring(s, substring):
    pattern = re.compile(r'\n*\s*' + re.escape(substring) + r'\s*\n*', re.IGNORECASE)
    modified_s = re.sub(pattern, '\n', s)
    return modified_s.strip() 
    
def retain_after_last_substring(s, substring):
    index = s.rfind(substring)
    if index != -1 and index + len(substring) < len(s):
        return s[index + len(substring):]
    else:
        return s
    
def get_retriever(config):
    r"""Automatically select retriever class based on config's retrieval method

    Args:
        config (dict): configuration with 'retrieval_method' key

    Returns:
        Retriever: retriever instance
    """
    if config["retrieval_method"] == "bm25":
        return getattr(importlib.import_module("retriever.retriever"), "BM25Retriever")(config)
    else:
        return getattr(importlib.import_module("retriever.retriever"), "DenseRetriever")(config)