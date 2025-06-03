import torch
from transformers import  AutoTokenizer, AutoModelForCausalLM

from .Model import Model
import os
import signal

def handle_timeout(sig, frame):
    raise TimeoutError('took too long')
signal.signal(signal.SIGALRM, handle_timeout)

class Llama(Model):
    def __init__(self, config, device = "cuda:0"):
        super().__init__(config)
        self.max_output_tokens = int(config["params"]["max_output_tokens"])

        api_pos = int(config["api_key_info"]["api_key_use"])
        hf_token = config["api_key_info"]["api_keys"][api_pos]
        self.tokenizer = AutoTokenizer.from_pretrained(self.name, use_auth_token=hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=device,
            use_auth_token=hf_token
        )
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        torch.set_default_tensor_type(torch.cuda.HalfTensor)

    def query(self, msg, max_tokens=128000):
        messages = self.messages
        messages[1]["content"] = msg

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)
        attention_mask = torch.ones(input_ids.shape, device=self.model.device)
        try:
            signal.alarm(60)

            output_tokens = self.model.generate(
                input_ids,
                max_length=max_tokens,
                attention_mask=attention_mask,
                eos_token_id=self.terminators,
                top_k=50,
                do_sample=False
            )
            signal.alarm(0)
        except TimeoutError as exc:
            print("time out")
            return("time out")
        # Decode the generated tokens back to text
        result = self.tokenizer.decode(output_tokens[0][input_ids.shape[-1]:], skip_special_tokens=True)
        return result

    def get_prompt_length(self,msg):
        messages = self.messages
        messages[1]["content"] = msg
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)
        return len(input_ids[0])
    def cut_context(self,msg,max_length):
        tokens = self.tokenizer.encode(msg, add_special_tokens=True)

        # Truncate the tokens to a maximum length
        truncated_tokens = tokens[:max_length]

        # Decode the truncated tokens back to text
        truncated_text = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        return truncated_text