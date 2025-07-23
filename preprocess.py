from __future__ import annotations

import random
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
from util import read_json,extract_labels,build_wordpiece_maps,flatten_sents
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

train_file ='/media/melissa/EXTERNAL_USB/Re-DocRED/data/train_revised.json'
pre_train_file ='/media/melissa/EXTERNAL_USB/DocRED/train_distant.json'
dev_file ='/media/melissa/EXTERNAL_USB/Re-DocRED/data/dev_revised.json'
test_file ='/media/melissa/EXTERNAL_USB/Re-DocRED/data/train_revised.json'


max_len =512 #when this is set as 128 or etc we lost data
out_dir ='/media/melissa/EXTERNAL_USB/docred_output'
rel2id_path ='/media/melissa/EXTERNAL_USB/DocRED/DocRED_baseline_metadata/rel2id.json'
tokenizer_name ="bert-base-cased"

# goal is emotion detection not classification so we also need 1:3, 1:5 ... ratio data that has no relation
# i chose 1:3 so 0.91, i also tested with 0.80(na:457_030) and 0.95
train_undersample_ratio = 0.91  #based on total count of possible pair of Na and gold label ratio
pre_train_undersample_ratio = 0.96  #based on total count of possible pair of Na and gold label ratio
na_label='Na' # rel2id no relation tag



@dataclass
class DocumentFeatures:
    """Container for a single processed document."""

    title: str
    input_ids: List[int]
    attention_mask: List[int]
    sentence_spans: List[Tuple[int, int]]
    entity_spans: List[List[Tuple[int, int]]]
    labels: List[Tuple[int, int, int]]  # (h, t, r_id)

    def to_dict(self) -> Dict:
        return asdict(self)


def process_document(example: Dict, tokenizer, max_len: int, rel2id: Dict[str, int], is_training=bool,split_name=str):

    title = example.get("title", "")
    sents: List[List[str]] = example["sents"]
    tokens, sent_offsets = flatten_sents(sents)

    # hhuggingFace tokenisation 
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        padding=False,
        max_length=max_len,
        return_attention_mask=True,
    )

    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    word_ids = encoding.word_ids()

    wp_start, wp_end = build_wordpiece_maps(word_ids)

    # sentence spans 
    sentence_spans: List[Tuple[int, int]] = []
    for i, sent_start_word in enumerate(sent_offsets):
        # last word index in this sentence
        if i + 1 < len(sent_offsets):
            sent_end_word = sent_offsets[i + 1] - 1
        else:
            sent_end_word = len(tokens) - 1
        # if truncated & last word lost, break.
        if sent_start_word not in wp_start or sent_end_word not in wp_end:
            break
        sentence_spans.append((wp_start[sent_start_word], wp_end[sent_end_word]))

    # entity span mapping
    entity_spans: List[List[Tuple[int, int]]] = []
    for mentions in example["vertexSet"]:
        mention_wp_spans: List[Tuple[int, int]] = []
        for m in mentions:
            m_start_word = m["pos"][0] if isinstance(m["pos"], list) else m["pos"]  # just in case
            m_end_word = m["pos"][1] if isinstance(m["pos"], list) else m["pos"]
            if (
                m_start_word in wp_start and m_end_word in wp_end and wp_start[m_start_word] < max_len - 1
            ):
                mention_wp_spans.append((wp_start[m_start_word], wp_end[m_end_word]))
        entity_spans.append(mention_wp_spans)

    # gold labels 
    labels = extract_labels(example, rel2id)


    # Random undersampling
    # For training only: undersample NA labels
    undersample_ratio = train_undersample_ratio
    if split_name == "pretrain":
        undersample_ratio = pre_train_undersample_ratio
    if split_name in ["train", "pretrain"]:
        
        positive_pairs = set((h, t) for h, t, _ in labels)

        num_entities = len(example['vertexSet']) #how many entities
        all_pairs = [(h, t) for h in range(num_entities) for t in range(num_entities) if h != t] # all possible label

        # for each negative pair, withhold it with undersample probability
        for h, t in all_pairs:
            if (h, t) not in positive_pairs and random.random() > (1 - undersample_ratio):
                continue  # skip adding this as negative

            # if positive, its already in labels; if negative, add 
            if (h, t) not in positive_pairs:
                labels.append((h, t, rel2id[na_label]))

    

    return DocumentFeatures(
        title=title,
        input_ids=input_ids,
        attention_mask=attention_mask,
        sentence_spans=sentence_spans,
        entity_spans=entity_spans,
        labels=labels,
    )

 

def build_file(file_path: str, split_name: str, tokenizer, rel2id: Dict[str, int]):

    examples: List[DocumentFeatures] = []

    data = read_json(file_path)

    for ex in tqdm(data, desc=f"{split_name}:{os.path.basename(file_path)}"):
        features = process_document(ex, tokenizer, max_len, rel2id,split_name)
        examples.append(features)

    torch.save([ex.to_dict() for ex in examples], os.path.join(out_dir, f"{split_name}.pt"))
    print(f"Saved {split_name} --> {os.path.join(out_dir, f'{split_name}.pt')}  ({len(examples)} docs)")





def preprocess():

    os.makedirs(out_dir, exist_ok=True)
    rel2id = read_json(rel2id_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    
    build_file(train_file, "train", tokenizer,  rel2id)   
    build_file(dev_file, "dev", tokenizer,  rel2id)
    build_file(test_file, "test", tokenizer,  rel2id)
    build_file(pre_train_file, "pretrain", tokenizer,  rel2id)


   





if __name__ == "__main__":
    preprocess()

