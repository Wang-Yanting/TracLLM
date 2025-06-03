import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .Model import Model
class HF_model(Model):
    def __init__(self, config, device="cuda:0"):
        super().__init__(config)
        self.max_output_tokens = int(config["params"]["max_output_tokens"])

        api_pos = int(config["api_key_info"]["api_key_use"])
        hf_token = config["api_key_info"]["api_keys"][api_pos]
        self.tokenizer = AutoTokenizer.from_pretrained(self.name, use_auth_token=hf_token, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            use_auth_token=hf_token,
            trust_remote_code=True
        )


    def query(self, msg, max_tokens=128000):    
        messages = self.messages
        messages[1]["content"] = msg
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=self.max_output_tokens,
            temperature=self.temperature
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def get_prompt_length(self,msg):
        messages = self.messages
        messages[1]["content"] = msg
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)
        return len(input_ids[0])

    def cut_context(self, msg, max_length):
        tokens = self.tokenizer.encode(msg, add_special_tokens=True)
        truncated_tokens = tokens[:max_length]
        truncated_text = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        return truncated_text