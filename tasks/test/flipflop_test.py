import os
import sys
import time
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

APP_ROOT = os.getenv("APP_ROOT", "/app")

from utilities.package_install import install

install("datasets")

import torch
import datasets

dataset = datasets.load_dataset('synthseq/flipflop', cache_dir="/app/cache/datasets")

def tokenize_batch(batch):
    mapping = {'w': 0, 'r': 1, 'i': 2, '0': 3, '1': 4}
    tokenized_batch = [[mapping[char] for char in s] for s in batch['text']]
    return {
        'text': batch['text'],
        'tokens': torch.tensor(tokenized_batch, dtype=torch.int64)
    }

dataset.set_transform(tokenize_batch)

for i in range(5):
    print(dataset['train'][i]['text'])  # e.g. 'w1i1w0i0 ...'
    print(dataset['train'][i]['tokens'])  # e.g. tensor([0, 4, 2, 4, 0, 3, 2, 3

# dataset['train'], dataset['val'], dataset['val_dense'], dataset['val_sparse']