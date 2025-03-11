import os
import openai
import torch
import transformers
os.environ["OPENAI_API_KEY"] = 'sk-proj-xxx'

class LLMChat():
    def __init__(self, model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'):
        super(LLMChat, self).__init__()
        self.model_name = model_name
        if model_name.startswith('gpt'):
            self.client = openai.OpenAI()
        else:
            self.pipeline = transformers.pipeline(
                "text-generation",
                model=model_name,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device="cuda",
            )

    def generate(self, messages, max_new_tokens = 500):
        if self.model_name.startswith('gpt'):
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0,
            )
            return response.choices[0].message.content
        else:
            outputs = self.pipeline(
                messages,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
            return outputs[0]["generated_text"][-1]
