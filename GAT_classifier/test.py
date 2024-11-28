from UFAChecker import get_test_loader
from UFAChecker import GATClassifier
from torch import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np


def test(test_loader=None):
    if test_loader == None:
        test_loader = get_test_loader()
    work_dir = "/home/aibot/workspace/SquiDroidAgent/MAGIC/0707result"
    device = 'cuda:0'
    model = torch.load(f'{work_dir}/0.9820.mdl').to(device)

    all_preds = []
    all_labels = []

    for data in tqdm(test_loader):
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
            predicted = out.max(dim=1)[1]
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    print("Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    return all_preds, all_labels

def test_single_case():
    test_loader = get_test_loader(flag_single=True,flag_hotload=False, pfn="/home/aibot/workspace/SquiDroidAgent/gensetg/data/test_case_adjacency_21_35_48.csv")
    test(test_loader)

def main():
    test_single_case()

if __name__ == "__main__":
    main()