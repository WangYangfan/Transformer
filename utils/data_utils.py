import os
import torch
from torch.utils.data import DataLoader

from transformers import BertTokenizer
from datasets import DatasetDict, load_from_disk

from loguru import logger
from typing import Dict, Tuple, Any

config = None
tokenizer_zh = None
tokenizer_en = None

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

def dataloader_init(config_init: Dict[str, Any]) -> Tuple[DataLoader]:
    global config, tokenizer_zh, tokenizer_en
    config = config_init
    tokenizer_zh = BertTokenizer.from_pretrained(config['checkpoint_zh'])
    tokenizer_en = BertTokenizer.from_pretrained(config['checkpoint_en'])

    """ load dataset """
    dataset = load_from_disk(config['dataset_path'])
    if config['num_shards'] is not None:
        dataset['train'] = dataset['train'].shard(num_shards=config['num_shards'], index=0)
        dataset['validation'] = dataset['validation'].shard(num_shards=config['num_shards'], index=0)
        dataset['test'] = dataset['test'].shard(num_shards=config['num_shards'], index=0)
    dataset_tokenized = dataset.map(
        preprocess_dataset,
        batched=True,
        load_from_cache_file=True,
        cache_file_names=_get_cache_file_names(dataset),
    )
    dataset_tokenized.set_format('pt')
    logger.info(f"💾 train: {len(dataset['train'])}, valid: {len(dataset['validation'])}, test: {len(dataset['test'])}")

    """ collator """
    train_dataloader = DataLoader(
        dataset=dataset_tokenized['train'],
        shuffle=True,
        collate_fn=collator_dataset,
        batch_size=config['batch_size']
    )
    valid_dataloader = DataLoader(
        dataset=dataset_tokenized['validation'],
        shuffle=True,
        collate_fn=collator_dataset,
        batch_size=config['batch_size']
    )
    test_dataloader = DataLoader(
        dataset=dataset_tokenized['test'],
        shuffle=True,
        collate_fn=collator_dataset,
        batch_size=config['batch_size']
    )

    return train_dataloader, valid_dataloader, test_dataloader
