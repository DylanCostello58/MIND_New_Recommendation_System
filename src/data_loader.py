import numpy as np
from collections import Counter
import nltk
import pandas as pd
nltk.download('punkt')
nltk.download('punkt_tab')

class NewsTokenizer:
    def __init__(self, max_title_len=30, min_word_freq=2):
        self.max_title_len = max_title_len
        self.min_word_freq = min_word_freq
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}


    def build_vocab(self, titles):
        word_counts = Counter()
        for title in titles:
            tokens = nltk.word_tokenize(title.lower())
            word_counts.update(tokens)
        for word, count in word_counts.items():
            if count >= self.min_word_freq:
                self.word2idx[word] = len(self.word2idx)
        print(f'Vocabulary size: {len(self.word2idx)}')


    def encode_title(self, title):
        tokens = nltk.word_tokenize(title.lower())
        indices = [self.word2idx.get(t, 1) for t in tokens]
        # Pad or truncate
        if len(indices) < self.max_title_len:
            indices += [0] * (self.max_title_len - len(indices))
        else:
            indices = indices[:self.max_title_len]
        return indices


def load_glove(glove_path, word2idx, embed_dim=300):
    embedding_matrix = np.random.normal(
        size=(len(word2idx), embed_dim)).astype('float32') * 0.1
    embedding_matrix[0] = 0  # PAD vector
    found = 0
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            if word in word2idx:
                embedding_matrix[word2idx[word]] =  np.array(parts[1:], dtype='float32')
                found += 1
    print(f'Found {found}/{len(word2idx)} words in GloVe')
    return embedding_matrix


def parse_behaviors(behaviors_df, news_encoded, neg_k=4):
    samples = []
    for _, row in behaviors_df.iterrows():
        history = row['history'].split() if pd.notna(row['history']) else []
        history_encoded = [news_encoded[nid]
                           for nid in history if nid in news_encoded]
        impressions = row['impressions'].split()
        pos = [imp.split('-')[0] for imp in impressions
               if imp.endswith('-1')]
        neg = [imp.split('-')[0] for imp in impressions
               if imp.endswith('-0')]
        for p in pos:
            sampled_neg = np.random.choice(
                neg, size=min(neg_k, len(neg)), replace=False)
            candidates = [p] + list(sampled_neg)
            labels = [1] + [0] * len(sampled_neg)
            samples.append({
                'history': history_encoded[-50:],  # last 50
                'candidates': [news_encoded[c]
                               for c in candidates
                               if c in news_encoded],
                'labels': labels
            })
    return samples
