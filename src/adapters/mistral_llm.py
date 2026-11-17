"""LLM local Mistral-7B cuantizado en 4 bits (bitsandbytes). Pensado para la T4 de Colab.

Mistral-7B-Instruct-v0.3 es un modelo restringido en Hugging Face: hay que aceptar sus términos con
la cuenta dueña del token y exponer ese token como la variable de entorno HF_TOKEN.
"""
from __future__ import annotations


class MistralLocalLLM:
    def __init__(self, model_id: str = "mistralai/Mistral-7B-Instruct-v0.3", max_new_tokens: int = 512,
                 stop_strings: tuple[str, ...] = ("Observation:",)) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.float16)
        try:
            self._tok = AutoTokenizer.from_pretrained(model_id)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id, quantization_config=quant, device_map="auto"
            )
        except OSError as e:
            raise RuntimeError(
                f"No se pudo cargar {model_id}. Si es un modelo restringido, acepta sus términos en "
                f"Hugging Face y define HF_TOKEN. Detalle: {e}"
            ) from e
        self._model_id = model_id
        self._max_new_tokens = max_new_tokens
        # El modelo suele inventar la Observation que sigue a una Action; cortar ahí ahorra tokens.
        self._stop_strings = list(stop_strings)

    @property
    def model(self) -> str:
        return f"{self._model_id} (4 bits)"

    def complete(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        inputs = self._tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self._model.device)
        out = self._model.generate(
            **inputs, max_new_tokens=self._max_new_tokens, do_sample=False,
            stop_strings=self._stop_strings, tokenizer=self._tok,
            pad_token_id=self._tok.eos_token_id,
        )
        new_tokens = out[0][inputs["input_ids"].shape[-1]:]
        return self._tok.decode(new_tokens, skip_special_tokens=True)
