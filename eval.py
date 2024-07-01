import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from accelerate import Accelerator
from transformers import BertTokenizer
from datasets import load_from_disk, DatasetDict
import evaluate
from loguru import logger
import argparse
from tqdm.auto import tqdm

import yaml
from typing import Dict, Any

from utils import seed_environment
from model import Transformer, model_init

def _get_cache_file_names(dataset: DatasetDict):
    return {
        k: os.path.join(
            config['cache_path'], "cache_file_{}_dataset_shards_{}".format(
                str(k), 
                config['num_shards'], 
            )
        ) for k in dataset
    }

def preprocess_dataset(examples: Dict[str, Any]) -> Dict[str, Any]:
    inputs = {}
    en_tokenized, zh_tokenized = [], []
    for en, zh in zip(examples['en'], examples['zh']):
        en_td = tokenizer_en(
            en,
            max_length=config['max_length'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        zh_td = tokenizer_zh(
            zh,
            max_length=config['max_length'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        en_tokenized.append(en_td)
        zh_tokenized.append(zh_td)
    inputs['en'] = en_tokenized
    inputs['zh'] = zh_tokenized
    return inputs

def collator_dataset(batch) -> Dict[str, Any]:
    batch_zh, batch_en = {}, {}
    batch_zh['input_ids'] = torch.cat([x['zh']['input_ids'] for x in batch], dim=0)
    batch_zh['attention_mask'] = torch.cat([x['zh']['attention_mask'] for x in batch], dim=0)
    # batch_zh['token_type_ids'] = torch.cat([x['zh']['token_type_ids'] for x in batch], dim=0)

    batch_en['input_ids'] = torch.cat([x['en']['input_ids'] for x in batch], dim=0)
    batch_en['attention_mask'] = torch.cat([x['en']['attention_mask'] for x in batch], dim=0)
    # batch_en['token_type_ids'] = torch.cat([x['en']['token_type_ids'] for x in batch], dim=0)
    return batch_zh, batch_en

def eval(model: Transformer, dataloader: DataLoader):
    model.eval()
    num_steps = len(dataloader)

    progress_bar = tqdm(range(num_steps), desc="🌟 Eval", unit="step")
    refs = []
    greedy_preds = []
    beam_preds = []
    for batch in dataloader:
        batch_zh, batch_en = batch
        ref_ids = batch_zh['input_ids'].tolist()
        greedy_ids = model.greedy_decode(batch_en)
        beam_ids = model.beam_search_decode(batch_en)

        # ref_gathered = accelerator.gather(ref_ids)
        # greedy_gathered = accelerator.gather(greedy_ids)
        # beam_gathered = accelerator.gather(beam_ids)

        decoded_refs = tokenizer_zh.batch_decode(ref_ids, skip_special_tokens=True)
        decoded_greedy = tokenizer_zh.batch_decode(greedy_ids, skip_special_tokens=True)
        decoded_beam = tokenizer_zh.batch_decode(beam_ids, skip_special_tokens=True)
        
        refs += decoded_refs
        greedy_preds += decoded_greedy
        beam_preds += decoded_beam

        progress_bar.update(1)

    refs = [[_] for _ in refs]

    scores = {
        'greedy': bleu_metric.compute(predictions=greedy_preds, references=refs, tokenize='zh')['score'],
        'beam': bleu_metric.compute(predictions=beam_preds, references=refs, tokenize='zh')['score'],
    }
    return scores

if __name__ == '__main__':
    """ Setup with config """
    logger.info(f'>>> looding config ...')
    with open('config.yml', 'r') as f:
        config: dict = yaml.safe_load(f)

    seed_environment(config['seed'])
    accelerator = Accelerator()
    config['device'] = accelerator.device

    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_dataset_path', type=str, default=config['dataset_path'])
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--num_shards', type=int, default=None)
    args = parser.parse_args()

    config['eval_dataset_path'] = args.eval_dataset_path
    config['model_path'] = args.model_path
    config['num_shards'] = args.num_shards
    assert config['model_path'] is not None

    for k, v in vars(args).items():
        logger.info(f'🧊 {k}: {v}, {type(v)}')


    tokenizer_zh = BertTokenizer.from_pretrained(config['checkpoint_zh'])
    tokenizer_en = BertTokenizer.from_pretrained(config['checkpoint_en'])
    eval_dataset = load_from_disk(config['dataset_path'])
    if config['num_shards'] is not None:
        eval_dataset['validation'] = eval_dataset['validation'].shard(num_shards=config['num_shards'], index=0)
        eval_dataset['test'] = eval_dataset['test'].shard(num_shards=config['num_shards'], index=0)
    for name in [str(_) for _ in eval_dataset if str(_) not in ('validation', 'test')]:
        del eval_dataset[name]

    eval_dataset_tokenized = eval_dataset.map(
        preprocess_dataset,
        batched=True,
        load_from_cache_file=True,
        cache_file_names=_get_cache_file_names(eval_dataset),
    )
    eval_dataset_tokenized.set_format('pt')
    logger.info("💾 valid: {}, test: {}".format(len(eval_dataset['validation']), len(eval_dataset['test'])))

    valid_dataloader = DataLoader(
        dataset=eval_dataset_tokenized['validation'],
        shuffle=True,
        collate_fn=collator_dataset,
        batch_size=config['batch_size']
    )
    test_dataloader = DataLoader(
        dataset=eval_dataset_tokenized['test'],
        shuffle=True,
        collate_fn=collator_dataset,
        batch_size=config['batch_size']
    )

    model: Transformer = model_init(config_init=config)
    # _model = SimcseModel(model_config=config)

    model, valid_dataloader, test_dataloader = accelerator.prepare(
        model, valid_dataloader, test_dataloader
    )
    bleu_metric = evaluate.load("./metrics/sacrebleu.py")

    accelerator.load_state(config['model_path'])
    model = accelerator.unwrap_model(model)

    for name, param in model.named_parameters():
        param.requires_grad = False

    # valid_score = eval(model, valid_dataloader)
    test_scores = eval(model, test_dataloader)

    # logger.info(f'(valid) score: {valid_score}')
    # logger.info(f'(test) score: {test_score}')       
    logger.info(f'(test) score: {str(test_scores)}')                         
