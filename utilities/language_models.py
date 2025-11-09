from .package_install import install
import os
import torch
from typing import List
import json
import time
import logging

install("pydantic")
from pydantic import BaseModel

class Chat_Message(BaseModel):
    role: str
    content: str

class Chat(BaseModel):
    messages: List[Chat_Message] = []

    def __add__(self, message: Chat_Message):
        concatenated_messages = self.messages + [message]
        return Chat(messages=concatenated_messages)

    def insert(self, index: int, message: Chat_Message):
        self.messages.insert(index, message)

    def append(self, message: Chat_Message):
        self.messages.append(message)

    def serialize(self):
        return [{"role": message.role, "content": message.content} for message in self.messages]
    
    @staticmethod
    def deserialize(json_obj):
        return Chat(messages=[Chat_Message(**message) for message in json_obj["messages"]])


deployment_type = os.getenv("PROJECT_LM_DEPLOYMENT", "cloud-api-litellm")

if deployment_type == "cloud-api-litellm":
    install("litellm")
    import litellm
    from litellm import text_completion, completion, embedding

    provider_api_key = os.getenv("PROJECT_LM_API_KEY", None)

    class Language_Model:
        def __init__(self, max_length=1024, top_p=0.95, temperature=0.6):
            self.model = os.getenv("PROJECT_LM_MODEL")
            self.max_length = max_length
            self.top_p = top_p
            self.temperature = temperature

        def complete_text(self, user_prompt):
            response = text_completion(
                model=self.model, 
                prompt=user_prompt,
                max_tokens=self.max_length,
                top_p=self.top_p,
                temperature=self.temperature,
                stop=["\n"],
                api_key=provider_api_key
            )
            text = response.choices[0].text
            return text
        
        def complete_chat(self, chat: Chat):
            messages = chat.serialize()
            response = completion(
                model=self.model, 
                messages=messages,
                max_tokens=self.max_length,
                top_p=self.top_p,
                temperature=self.temperature,
                api_key=provider_api_key
            )
            text = response.choices[0].message.content
            return text

elif deployment_type == "cloud-api-runpod":
    install("requests")
    import requests

    runpod_api_key = os.getenv("PROJECT_LM_API_KEY", None)
    model_name = os.getenv("PROJECT_LM_MODEL")
    runpod_endpoint = f"https://api.runpod.ai/v2/{model_name}/run"
    runpod_status_endpoint = f"https://api.runpod.ai/v2/{model_name}/status/"

    def wait_for_result(response):
        poll_interval = 1
        try:
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            job = response.json()
            job_id = job["id"]

            headers = {
                "Authorization": f"Bearer {runpod_api_key}",
                "Content-Type": "application/json",
            }
            for i in range(60):
                status_response = requests.get(runpod_status_endpoint + job_id, headers=headers)
                status_response.raise_for_status()
                job_status = status_response.json()

                if job_status["status"] == "COMPLETED":
                    return job_status["output"]
                elif job_status["status"] == "FAILED":
                    raise Exception(f"RunPod job failed: {job_status['error']}")
                elif job_status["status"] == "IN_QUEUE" or job_status["status"] == "IN_PROGRESS":
                    logging.info(f"Job status: {job_status['status']}. Polling again in {poll_interval} seconds.")
                    time.sleep(poll_interval)
                else:
                    raise Exception(f"Unexpected job status: {job_status['status']}")
                
                poll_interval = min(2 * poll_interval, 10)

        except requests.exceptions.RequestException as e:
            raise Exception(f"Request error: {e}")
        except json.JSONDecodeError as e:
            raise Exception(f"JSON decode error: {e}")

        raise Exception(f"Job did not complete in time. Last status: {job_status['status']}")


    class Language_Model:
        def __init__(self, max_length=1024, top_p=0.95, temperature=0.6):
            self.sampling_params = {
                "max_tokens": max_length,
                "top_p": top_p,
                "temperature": temperature,
                "stop": ["\n"]
            }

        def complete_text(self, user_prompt):
            response = requests.post(runpod_endpoint, headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {runpod_api_key}"
            }, json={
                "input": {"prompt": user_prompt},
                "sampling_params": self.sampling_params
            })
            output = wait_for_result(response)
            return output[0]['choices'][0]['tokens'][0]
        

        def complete_chat(self, chat: Chat):
            messages = chat.serialize()
            response = requests.post(runpod_endpoint, headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {runpod_api_key}"
            }, json={
                "input": {"messages": messages},
                "sampling_params": self.sampling_params
            })
            output = wait_for_result(response)
            text = output[0]['choices'][0]['tokens'][0]
            return text

elif deployment_type == "local-hf":

    APP_ROOT = os.getenv("APP_ROOT", "/app")
    os.environ['HF_HOME'] = f"{APP_ROOT}/cache/hf_home"
    install("transformers")

    from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline

    class Language_Model:

        def __init__(self, max_length=1024, top_p=0.95, temperature=0.6):
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model_name = os.getenv("PROJECT_LM_MODEL")
            self.is_gptq = "gptq" in self.model_name.lower()
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                trust_remote_code=True,
                torch_dtype=torch.float16
            ).to(self.device)

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True,
                use_fast=True
            )

            self.generator = TextGenerationPipeline(
                model=self.model,
                tokenizer=self.tokenizer
            )

            self.generation_args = {
                "max_new_tokens": max_length,
                "temperature": temperature + 0.01,
                "top_p": top_p,
                "do_sample": True,
                "stop_strings": ["\n"],
                "tokenizer": self.tokenizer,
            }
        
        def complete_chat(self, chat: Chat):
            messages = chat.serialize()
            if self.is_gptq:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                return self.tokenizer.decode(self.model.generate(**self.tokenizer(text, return_tensors="pt").to(self.model.device), **self.generation_args)[0])
            else:
                return self.generator(messages, return_full_text=False, **self.generation_args)[0]['generated_text']
            
        def complete_text(self, user_prompt:str):
            if self.is_gptq:
                return self.tokenizer.decode(self.model.generate(**self.tokenizer(user_prompt, return_tensors="pt").to(self.model.device), **self.generation_args)[0])
            else:
                return self.generator(user_prompt, return_full_text=False, **self.generation_args)[0]['generated_text']
        