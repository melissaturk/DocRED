import torch
from torch.utils.data import DataLoader
from model import BertForDocRE
from util import DocREDTorchDataset, collate_fn,write_log,read_json
import json
from datetime import datetime
import os



base_path = '/media/melissa/EXTERNAL_USB/docred_output' 

error_path = os.path.join(base_path, "error_eval_log.txt")
pred_log_path = os.path.join(base_path, "predictions.jsonl")
 
model_path=os.path.join(base_path, "bert_docre.pt")
threshold_path=os.path.join(base_path, "thresholds.json")
test_path=os.path.join(base_path, "test.pt")
rel2id_path ='/media/melissa/EXTERNAL_USB/DocRED/DocRED_baseline_metadata/rel2id.json'

threshold=0.3 #former val


 

def predict_relations(model, dataloader, device='cuda'):
    model.eval()
    predictions = []

    best_thresholds = read_json(threshold_path)
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            entity_spans = batch['entity_spans']
            titles = batch['title']

            logits_batch = model(input_ids, attention_mask, entity_spans)

            for i, doc_logits in enumerate(logits_batch):
                doc_preds = []
                for h, t, logits in doc_logits:
                    scores = torch.sigmoid(logits)
                    #pred_rels = (scores > threshold).nonzero(as_tuple=True)[0].tolist()
                    #for r in pred_rels:
                    for r in (scores > 0.05).nonzero(as_tuple=True)[0].tolist():
                        thresh = best_thresholds.get(str(r), 0.1)

                        if scores[r] > thresh:
                            doc_preds.append([h, t, r])
                            
                predictions.append({
                    "title": titles[i],
                    "labels": doc_preds
                })
                print(titles[i],doc_preds)

    return predictions


 
def evaluate(num_relations=97, batch_size=4, device='cuda'):
    model = BertForDocRE(num_relations=num_relations).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    dataset = DocREDTorchDataset(test_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    
  
    preds = predict_relations(model, dataloader, device=device)

    
    with open(pred_log_path, "w", encoding="utf-8") as f:
        for doc_pred in preds:
            f.write(json.dumps(doc_pred) + "\n")
    print(f"Predictions saved to {pred_log_path}")


 
    



def eval():
    try:
        evaluate()     
    except Exception as e:
        print(f"Error encountered: {e}")
        msg=f"{datetime.now()} :Error encountered: {e}"
        write_log(msg,error_path)


if __name__ == "__main__":
    eval()
