import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('..')
from src.model import NRMSModel


class MINDDataset(Dataset):
    def __init__(self, samples, max_history=50, max_title_len=30, neg_k=4):
        self.samples = samples
        self.max_history = max_history
        self.max_title_len = max_title_len
        self.neg_k = neg_k

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Pad or truncate history
        history = list(sample['history'][-self.max_history:])
        hist_mask = [1] * len(history)
        while len(history) < self.max_history:
            history.append([0] * self.max_title_len)
            hist_mask.append(0)

        # Fix candidates and labels to exactly neg_k+1
        candidates = list(sample['candidates'])
        labels = list(sample['labels'])

        candidates = candidates[:self.neg_k + 1]
        labels = labels[:self.neg_k + 1]

        while len(candidates) < self.neg_k + 1:
            candidates.append([0] * self.max_title_len)
            labels.append(0)

        candidates = [c[:self.max_title_len] + [0] * max(0, self.max_title_len - len(c))
                      for c in candidates]

        return {
            'history': torch.tensor(history, dtype=torch.long),
            'candidates': torch.tensor(candidates, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'hist_mask': torch.tensor(hist_mask, dtype=torch.long)
        }


def train(model, train_loader, optimizer, criterion, device, epochs, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs('../results', exist_ok=True)
    model.train()
    epoch_losses = []

    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            history = batch['history'].to(device)
            candidates = batch['candidates'].to(device)
            labels = batch['labels'].to(device)
            hist_mask = batch['hist_mask'].to(device)

            optimizer.zero_grad()
            scores = model(history, candidates, hist_mask)

            target = torch.zeros(scores.size(0), dtype=torch.long).to(device)
            loss = criterion(scores, target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
        torch.save(model.state_dict(), f'{save_dir}/nrms_epoch{epoch+1}.pt')

    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), epoch_losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Loss Curve')
    plt.xticks(range(1, epochs + 1))
    plt.tight_layout()
    plt.savefig('../results/loss_curve.png')
    plt.show()
    print('Loss curve saved to results/loss_curve.png')

    return epoch_losses