import torch.nn as nn
from torch.amp import autocast
from datetime import datetime
import os
import joblib
import torch
import gc
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer,AutoModel
import random
import logging
import json
import numpy as np
from model import BertForDocRE
from util import DocREDTorchDataset, collate_fn,write_log,read_json
from collections import defaultdict
from sklearn.metrics import f1_score
import shutil


#base_path = '/home/mt24606/docred'
base_path = '/media/melissa/EXTERNAL_USB/docred_output' 

error_path = os.path.join(base_path, "error_log.txt")
train_log_path = os.path.join(base_path, "train_log.txt")
dev_log_path = os.path.join(base_path, "dev_log.txt")

train_path=os.path.join(base_path, "train.pt")
dev_path=os.path.join(base_path, "dev.pt")
best_thresholds = os.path.join(base_path, "thresholds.json")
threshold = 0.5  # for initial


model_path=os.path.join(base_path, "bert_docre.pt")


os.makedirs(base_path, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(base_path, 'training.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)



def compute_loss(pred_logits, gold_labels, num_relations, device):
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    for i, doc_logits in enumerate(pred_logits):
        #  gold tensor: shape (num_pairs, num_relations)
        label_dict = {(h, t): [] for h, t, _ in doc_logits}
        for h, t, r in gold_labels[i]:
            if (h, t) in label_dict:
                label_dict[(h, t)].append(r)

        gold_tensor = torch.zeros(len(doc_logits), num_relations, device=device)
        for j, (h, t, _) in enumerate(doc_logits):
            for r in label_dict.get((h, t), []):
                gold_tensor[j][r] = 1

        logits = torch.stack([x[2] for x in doc_logits])  # (num_pairs, num_relations)
        loss = criterion(logits, gold_tensor)
        total_loss += loss
    return total_loss / len(pred_logits)




def train_model(num_relations=97, epochs=2, batch_size=4, lr=2e-5, device='cpu'):
    train_data = DocREDTorchDataset(train_path)
    dev_data = DocREDTorchDataset(dev_path)

   
    # test small subset for quick testing  
    train_data = torch.utils.data.Subset(train_data, list(range(10)))
    dev_data = torch.utils.data.Subset(dev_data, list(range(10)))
    
     
 


    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    logging.info(f"train_loader size: {len(train_loader)}")

    model = BertForDocRE(num_relations=num_relations).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    results = {
        'time': datetime.now(),
        'train_loss': [],
        'dev_loss': []
    }

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        len_tl = len(train_loader)

        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            entity_spans = batch['entity_spans']
            labels = batch['labels']

            optimizer.zero_grad()
            pred_logits = model(input_ids, attention_mask, entity_spans)
            loss = compute_loss(pred_logits, labels, num_relations, device)
            

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if device == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()

                 
            logging.info(f"[Batch {step+1}/{len_tl}] Loss: {loss.item():.4f}")
            del input_ids, attention_mask, entity_spans, labels, pred_logits, loss

        avg_train_loss = total_loss / len(train_loader)
        results['train_loss'].append(avg_train_loss)
        logging.info(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}")

        # evaluation mode for validation


        TP, FP, FN = 0, 0, 0
        dev_loss = 0.0

        model.eval()
        dev_loss = 0.0
        all_scores = defaultdict(list)
        all_labels = defaultdict(list)
        best_f1_total = -1
        best_thresholds_epoch = 0

        with torch.no_grad():

            for batch in dev_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                entity_spans = batch['entity_spans']
                gold_labels = batch['labels']

                pred_logits = model(input_ids, attention_mask, entity_spans)
                loss = compute_loss(pred_logits, gold_labels, num_relations, device)
                dev_loss += loss.item()


                logging.info(f"[DEV Batch {step+1}/{len_tl}] Loss: {loss.item():.4f}")

             
                for i, doc_logits in enumerate(pred_logits):
                    gold_set = set(tuple(x) for x in gold_labels[i])

                    pred_set = set()
                    for h, t, logits in doc_logits:
                        probs = torch.sigmoid(logits)

                        for r in range(len(probs)):
                            all_scores[r].append(probs[r].item())
                                                        
                            gold_set = set((h, t, r) for h, t, r in gold_labels[i])
                            label = 1 if (h, t, r) in gold_set else 0
                            all_labels[r].append(label)
                        

                        '''
                        for r, p in enumerate(probs):
                            rel_thresh = best_thresholds.get(r, 0.1)
                            if p > rel_thresh:
                                pred_set.add((h, t, r))
                        

                        for r, p in enumerate(probs):
                            if p > threshold:
                                pred_set.add((h, t, r))'''

                    TP += len(gold_set & pred_set)
                    FP += len(pred_set - gold_set)
                    FN += len(gold_set - pred_set)


                del input_ids, attention_mask, entity_spans, gold_labels, pred_logits, loss

        
        precision = TP / (TP + FP + 1e-10)
        recall = TP / (TP + FN + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        avg_dev_loss = dev_loss / len(dev_loader)
        results['dev_loss'].append(avg_dev_loss)

        logging.info(f"[Epoch {epoch+1}] Dev Loss: {avg_dev_loss:.4f}")
        logging.info(f"[Epoch {epoch+1}] Dev Precision: {precision:.4f}")
        logging.info(f"[Epoch {epoch+1}] Dev Recall: {recall:.4f}")
        logging.info(f"[Epoch {epoch+1}] Dev F1: {f1:.4f}")
        
        
        # === save thresholds ===
        collect_threshold(all_scores,all_labels,epoch)

        if f1 > best_f1_total:
            best_f1_total = f1
            best_thresholds_epoch = epoch
            shutil.copyfile(
                os.path.join(base_path, f"thresholds_epoch_{epoch}.json"),
                os.path.join(base_path, "thresholds.json")
            )
        

    torch.save(model.state_dict(), model_path)
    logging.info("Model saved to bert_docre.pt")


def collect_threshold(all_scores, all_labels, epoch):
    best_thresholds = {}

    for r in all_scores:
        scores = np.array(all_scores[r])
        labels = np.array(all_labels[r])

        best_f1 = 0
        best_thresh = 0.5
        for thresh in np.arange(0.05, 0.95, 0.05):
            preds = (scores > thresh).astype(int)
            f1 = f1_score(labels, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        best_thresholds[r] = best_thresh
        logging.info(f"Relation {r}: Best threshold = {best_thresh:.2f}, F1 = {best_f1:.3f}")
    
    path = os.path.join(base_path, f"thresholds_epoch_{epoch}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(best_thresholds, f, indent=2)

    print(f"Best threshold saved for epoch {epoch}")





def main():
    try:
        train_model()
          
    except Exception as e:
        print(f"Error encountered: {e}")
        msg=f"{datetime.now()} :Error encountered: {e}"
        write_log(msg,error_path)


if __name__ == "__main__":
    main()
