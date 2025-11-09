import re
import numpy as np
import torch
import os
import json


class Text_Tokenizer:

    def __init__(self, max_vocab_size, device):
        self.max_vocab_size = max_vocab_size
        self.device = device
        self.id2word = ["<PAD>", "<UNK>"]
        self.word2id = {w: i for i, w in enumerate(self.id2word)}


    def save(self, path):
        tokenizer_path = os.path.join(path, "tokenizer_data.json")
        with open(tokenizer_path, "w") as f:
            tokenizer = {
                "max_vocab_size": self.max_vocab_size,
                "id2word": self.id2word,
                "word2id": self.word2id
            }
            json.dump(tokenizer, f, indent=4)
        

    def load(self, path):
        tokenizer_path = os.path.join(path, "tokenizer_data.json")
        if not os.path.exists(tokenizer_path):
            return False
        with open(tokenizer_path, "r") as f:
            tokenizer = json.load(f)
            self.max_vocab_size = tokenizer["max_vocab_size"]
            self.id2word = tokenizer["id2word"]
            self.word2id = tokenizer["word2id"]
        return True


    def _get_word_id(self, word):
        if word not in self.word2id:
            if len(self.word2id) >= self.max_vocab_size:
                return self.word2id["<UNK>"]

            self.id2word.append(word)
            self.word2id[word] = len(self.word2id)

        return self.word2id[word]


    def _tokenize(self, text):
        # Simple tokenizer: strip out all non-alphabetic characters.
        text_parts = re.findall(r"[a-zA-Z0-9\-]+", text)
        word_ids = list(map(self._get_word_id, text_parts))
        return word_ids
    

    def __len__(self):
        return len(self.id2word)
    

    def __call__(self, texts, stack=True):
        texts = list(map(self._tokenize, texts))

        if stack:
            max_len = max(len(l) for l in texts)
            padded = np.ones((len(texts), max_len)) * self.word2id["<PAD>"]

            for i, text in enumerate(texts):
                padded[i, :len(text)] = text

            padded_tensor = torch.from_numpy(padded).type(torch.long).to(self.device)
            return padded_tensor
        else:
            # return list of tensors instead
            return [torch.from_numpy(np.array(text)).type(torch.long).to(self.device) for text in texts]
