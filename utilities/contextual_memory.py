from .embedding_models import embed
import torch
import re


def chunk(paragraph: str, chunk_size, splitters):
    # first split paragraph, then try to accumulate parts to make each part's length <= chunk_size
    if isinstance(splitters, list):
        splitters = "|".join(splitters)
        new_splitter = splitters[0]
    else:
        new_splitter = splitters
    parts = re.split(splitters, paragraph)
    chunks = []
    chunk = ""
    for part in parts:
        if len(chunk) + len(part) > chunk_size:
            chunks.append(chunk)
            chunk = part
        else:
            chunk += new_splitter + part
    if chunk:
        chunks.append(chunk)
    return chunks



class Vector_Text_Dictionary:

    def __init__(self, paragraphs, metadata=None, max_chunk_size=None, splitters=None):
        self.paragraphs = paragraphs
        self.metadata = metadata

        self.embeddings = []
        self.indices = []
        for i, paragraph in enumerate(paragraphs):
            if max_chunk_size is None:
                embeddings = embed([paragraph])
            else:
                embeddings = embed(chunk(paragraph, chunk_size=max_chunk_size, splitters=splitters))
            self.embeddings.append(embeddings)
            self.indices.append(torch.ones(embeddings.shape[0], dtype=torch.int)*i)

        self.embeddings = torch.concat(self.embeddings, dim=0)
        self.indices = torch.concat(self.indices, dim=0)


    def match(self, query, k=1):
        query_embedding = embed(query)
        metric = torch.nn.functional.cosine_similarity(self.embeddings, torch.reshape(query_embedding, [1, -1]), dim=-1)
        # get top_k indices
        top_k_indices = torch.topk(metric, k=k, largest=True, sorted=True).indices
        # get top_k para_indices
        top_k_para_indices = self.indices[top_k_indices[0]]
        # change to python list
        return top_k_para_indices.tolist()
        

    def get_paragraph(self, para_index):
        return self.paragraphs[para_index]
    

    def get_metadata(self, para_index):
        return self.metadata[para_index]
    

class Hippocampus:

    def __init__(self):
        self.memory = {}


    def store(self, key, value):
        self.memory[key] = value


    def retrieve(self, key):
        return self.memory[key]