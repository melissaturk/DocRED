import torch
import torch.nn as nn
from transformers import AutoModel
from torch.amp import autocast

class BertForDocRE(nn.Module):
    def __init__(self, encoder_name='bert-base-cased', num_relations=97, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_size = self.encoder.config.hidden_size

        # Classifier takes [entity_h; entity_t] vector
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_relations)  # Multi-label logits (no softmax)
        )

    def forward(self, input_ids, attention_mask, entity_spans):


        #with autocast():
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # (B, L, H)

        batch_size = input_ids.size(0)
        relation_logits = []

        for b in range(batch_size):
            entity_reps = []
            for spans in entity_spans[b]:
                if not spans:
                    entity_reps.append(torch.zeros_like(last_hidden[b, 0]))
                else:
                    # Average all wordpieces of all mentions
                    pieces = [last_hidden[b, start:end+1].mean(dim=0) for (start, end) in spans]
                    entity_reps.append(torch.stack(pieces).mean(dim=0))

            pair_logits = []
            for h in range(len(entity_reps)):
                for t in range(len(entity_reps)):
                    if h == t:
                        continue
                    pair_vec = torch.cat([entity_reps[h], entity_reps[t]], dim=-1)
                    logits = self.classifier(pair_vec)
                    pair_logits.append((h, t, logits))  # Save indices too
            relation_logits.append(pair_logits)

        return relation_logits  # List[batch] → List[(h, t, logits)] per doc
