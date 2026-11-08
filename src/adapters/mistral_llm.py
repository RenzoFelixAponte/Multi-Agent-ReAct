"""LLM local Mistral-7B cuantizado en 4 bits (bitsandbytes). Pensado para la T4 de Colab."""
from __future__ import annotations


class MistralLocalLLM:
    def __init__(self, model_id: str = "mistralai/Mistral-7B-Instruct-v0.3", max_new_tokens: int = 512) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        self._tok = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=quant, device_map="auto"
        )
        self._max_new_tokens = max_new_tokens

    def complete(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        ids = self._tok.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
        ids = ids.to(self._model.device)
        out = self._model.generate(ids, max_new_tokens=self._max_new_tokens, do_sample=False)
        return self._tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)
