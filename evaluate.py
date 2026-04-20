from sklearn.metrics import roc_auc_score
import numpy as np
import torch

def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1][:k]
    gains = np.array(y_true)[order]
    discounts = np.log2(np.arange(len(gains)) + 2)
    return np.sum(gains / discounts)

def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    if best == 0: return 0.0
    return dcg_score(y_true, y_score, k) / best

def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_sorted = np.array(y_true)[order]
    for i, val in enumerate(y_sorted):
        if val == 1:
            return 1.0 / (i + 1)
    return 0.0

def evaluate(model, val_loader, device):
    model.eval()
    aucs, mrrs, ndcg5s, ndcg10s = [], [], [], []
    with torch.no_grad():
        for batch in val_loader:
            history = batch['history'].to(device)
            candidates = batch['candidates'].to(device)
            hist_mask = batch['hist_mask'].to(device)
            labels = batch['labels']
            scores = model(history, candidates, hist_mask).cpu().numpy()

            for i in range(len(labels)):
                y_true = labels[i].numpy() if torch.is_tensor(labels[i]) else np.array(labels[i])
                y_score = scores[i]
                # truncate y_true to match model output length
                y_true = y_true[:len(y_score)]
                if sum(y_true) == 0 or sum(y_true) == len(y_true):
                    continue
                aucs.append(roc_auc_score(y_true, y_score))
                mrrs.append(mrr_score(y_true, y_score))
                ndcg5s.append(ndcg_score(y_true, y_score, 5))
                ndcg10s.append(ndcg_score(y_true, y_score, 10))

    print(f'AUC:     {np.mean(aucs):.4f}')
    print(f'MRR:     {np.mean(mrrs):.4f}')
    print(f'nDCG@5:  {np.mean(ndcg5s):.4f}')
    print(f'nDCG@10: {np.mean(ndcg10s):.4f}')
    return {
        'AUC': np.mean(aucs), 'MRR': np.mean(mrrs),
        'nDCG@5': np.mean(ndcg5s), 'nDCG@10': np.mean(ndcg10s)
    }