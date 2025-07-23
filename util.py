from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset




def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_log(msg, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a') as output_file: 
        output_file.write(str(msg) + "\n")


def flatten_sents(sents: List[List[str]]):
    """Flatten nested token list & capture sentence start offsets."""
    tokens: List[str] = []
    sent_word_offsets: List[int] = []
    for sent in sents:
        sent_word_offsets.append(len(tokens))
        tokens.extend(sent)
    return tokens, sent_word_offsets


def build_wordpiece_maps(word_ids: List[int | None]):
    """Build mappings word_idx → (wp_start, wp_end) inclusive."""
    start_map: Dict[int, int] = {}
    end_map: Dict[int, int] = {}
    for wp_idx, w_idx in enumerate(word_ids):
        if w_idx is None:
            continue
        if w_idx not in start_map:
            start_map[w_idx] = wp_idx
        end_map[w_idx] = wp_idx
    return start_map, end_map


def extract_labels(example: Dict, rel2id: Dict[str, int]) -> List[Tuple[int, int, int]]:
    """Convert *labels* (and *labels2_annotator_id* if present) to id triples."""
    labels_output: List[Tuple[int, int, int]] = []
    raw_labels = example.get("labels", [])
    # Re‑DocRED sometimes stores a list per annotator under this key
    raw_labels += example.get("labels2_annotator_id", [])

    for label in raw_labels:
        rel_name = label["r"] if isinstance(label, dict) else label.get("r")
        labels_output.append((label["h"], label["t"], rel2id[rel_name]))
    return labels_output


class DocREDTorchDataset(Dataset):
    def __init__(self, data_path):
        self.data = torch.load(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    input_ids = [torch.tensor(x['input_ids']) for x in batch]
    attention_mask = [torch.tensor(x['attention_mask']) for x in batch]

    max_len = max(len(x) for x in input_ids)
    padded_input_ids = torch.stack([
        torch.cat([ids, torch.zeros(max_len - len(ids), dtype=torch.long)]) for ids in input_ids
    ])
    padded_attention = torch.stack([
        torch.cat([mask, torch.zeros(max_len - len(mask), dtype=torch.long)]) for mask in attention_mask
    ])

    # entity spans: keep as list of lists of tuples
    entity_spans = [x['entity_spans'] for x in batch]
    labels = [x['labels'] for x in batch]  # (h, t, r)
    titles =[x['title'] for x in batch]
    return {
        'input_ids': padded_input_ids,
        'attention_mask': padded_attention,
        'entity_spans': entity_spans,
        'labels': labels,
        'title' : titles

    }
