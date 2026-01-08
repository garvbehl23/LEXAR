from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class LexarGenerator:
    def __init__(self, model_name="google/flan-t5-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()

    def generate(self, prompt: str, max_tokens: int = 200):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens
            )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)
