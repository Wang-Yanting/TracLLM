from openai import OpenAI
from .Model import Model
import tiktoken
from transformers import AutoTokenizer
import time
class GPT(Model):
    def __init__(self, config):
        super().__init__(config)
        api_keys = config["api_key_info"]["api_keys"]
        api_pos = int(config["api_key_info"]["api_key_use"])
        assert (0 <= api_pos < len(api_keys)), "Please enter a valid API key to use"
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        self.client = OpenAI(api_key=api_keys[api_pos])
        self.llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.seed = 10

    def query(self, msg, max_tokens=128000):
        super().query(max_tokens)
        while True:
            try:
                completion = self.client.chat.completions.create(
                    model=self.name,
                    temperature=self.temperature,
                    max_tokens=self.max_output_tokens,
                    seed = self.seed,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": msg}
                    ],
                )
                response = completion.choices[0].message.content
                time.sleep(1)
                break
            except Exception as e:
                print(e)
                time.sleep(10)
        return response
    
    def get_prompt_length(self,msg):
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        num_tokens = len(encoding.encode(msg))
        return num_tokens
    
    def cut_context(self,msg,max_length):
        tokens = self.encoding.encode(msg)
        truncated_tokens = tokens[:max_length]
        truncated_text = self.encoding.decode(truncated_tokens)
        return truncated_text