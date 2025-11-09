from .package_install import install
import os
import torch

deployment_type = os.getenv("PROJECT_EMBEDDING_DEPLOYMENT", "cloud-api-litellm")

if deployment_type == "cloud-api-litellm":
    install("litellm")
    import litellm
    from litellm import embedding

    model = os.getenv("PROJECT_EMBEDDING_MODEL")    
    provider_api_key = os.getenv("PROJECT_EMBEDDING_API_KEY", None)

    def embed(text):
        # if a list then embed list
        if isinstance(text, list):
            response = embedding(
                model=model, 
                input=text,
                api_key=provider_api_key
            )
            return torch.tensor([embedding["embedding"] for embedding in response.data], dtype=torch.float32)
        else:
            response = embedding(
                model=model, 
                input=[text],
                api_key=provider_api_key
            )
            return torch.tensor(response.data[0]["embedding"], dtype=torch.float32)

elif deployment_type == "local-hf":

    APP_ROOT = os.getenv("APP_ROOT", "/app")
    os.environ['HF_HOME'] = f"{APP_ROOT}/cache/hf_home"
    install("transformers")

    from transformers import AutoTokenizer, AutoModel, pipeline

    model_name = os.getenv("PROJECT_EMBEDDING_MODEL")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    
    def embed(text):
        if isinstance(text, list):
            input_queries = text
        else:
            input_queries = [text]

        tokenized_queries = tokenizer(input_queries, padding=True, truncation=True, return_tensors='pt').to(device)

        with torch.no_grad():
            # Queries
            model_output = model(**tokenized_queries)
            # Perform pooling. granite-embedding-30m-english uses CLS Pooling
            query_embeddings = model_output[0][:, 0]

        # normalize the embeddings
        query_embeddings = torch.nn.functional.normalize(query_embeddings, dim=1)

        # if a list then embed list
        if isinstance(text, list):
            return query_embeddings
        else:
            return query_embeddings[0, :]