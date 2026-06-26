from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from transformers.modeling_outputs import CausalLMOutput
from transformers import GenerationConfig, PreTrainedTokenizerBase, PretrainedConfig

from .notagen_wrapper import (
    NotaGenPolicyWrapper,
    decode_flat_patch_ids,
    select_device,
)


class NotaGenAdapterConfig(PretrainedConfig):
    model_type = "notagen-adapter"

    def __init__(
        self,
        weights_path: str,
        precision: str = "bf16",
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, eos_token_id=eos_token_id, **kwargs)
        self.weights_path = weights_path
        self.precision = precision
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout


class NotaGenProcessingStub(PreTrainedTokenizerBase):
    model_input_names = ["input_ids"]

    def __init__(self):
        super().__init__(padding_side="left", truncation_side="left")
        self._pad_token = "<pad>"
        self._eos_token = "<eos>"
        self._bos_token = "<bos>"
        self.pad_token = self._pad_token
        self.eos_token = self._eos_token
        self.bos_token = self._bos_token

    @property
    def pad_token(self):
        return self._pad_token

    @property
    def eos_token(self):
        return self._eos_token

    @property
    def bos_token(self):
        return self._bos_token

    def __call__(self, text, *args, **kwargs):
        if isinstance(text, list):
            return {"input_ids": text}
        return {"input_ids": [text]}

    def batch_decode(self, sequences, *args, **kwargs):
        out = []
        for seq in sequences:
            if isinstance(seq, str):
                out.append(seq)
            else:
                out.append(decode_flat_patch_ids(list(seq)))
        return out

    def decode(self, token_ids, *args, **kwargs):
        if isinstance(token_ids, str):
            return token_ids
        return decode_flat_patch_ids(list(token_ids))

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        vocab = {0: "<pad>", 1: "<eos>"}
        if isinstance(ids, list):
            return [vocab.get(int(i), str(i)) for i in ids]
        return vocab.get(int(ids), str(ids))

    def convert_tokens_to_ids(self, tokens):
        vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 0}
        if isinstance(tokens, list):
            return [vocab.get(t, 0) for t in tokens]
        return vocab.get(tokens, 0)

    def get_vocab(self):
        return {"<pad>": 0, "<eos>": 1}


@dataclass
class NotaGenGeneratedOutput:
    generated_tokens: list[int]


class NotaGenHFAdapter(torch.nn.Module):
    """
    Thin adapter for the TRL continuous-batching path.

    This does not expose a standard token-level HF LM interface. It only exposes:
    - `generate_batch(prompt_texts, generation_config=...)`
    - `score_completions(prompt_completion_pairs)`

    That is the narrowest useful bridge from NotaGen into HF/TRL-style orchestration.
    """

    def __init__(
        self,
        weights_path: str,
        *,
        precision: str = "bf16",
        device: torch.device | None = None,
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.config = NotaGenAdapterConfig(
            weights_path=weights_path,
            precision=precision,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.device_ref = device or select_device()
        self.wrapper = NotaGenPolicyWrapper(weights_path=weights_path, device=self.device_ref, precision=precision)
        self.policy_model = self.wrapper.model
        self._configure_trainable_model()
        self._gradient_checkpointing = False
        self.generation_config = GenerationConfig(
            do_sample=True,
            temperature=1.0,
            top_k=8,
            top_p=0.95,
            max_new_tokens=256,
            pad_token_id=self.config.pad_token_id,
            eos_token_id=self.config.eos_token_id,
        )

    @property
    def device(self) -> torch.device:
        return self.device_ref

    def _configure_trainable_model(self):
        for param in self.policy_model.parameters():
            param.requires_grad = False

        if not self.config.use_lora:
            return

        lora_cfg = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            target_modules=["attn.c_attn", "attn.c_proj"],
        )
        self.policy_model.patch_level_decoder.base = get_peft_model(self.policy_model.patch_level_decoder.base, lora_cfg)
        self.policy_model.char_level_decoder.base = get_peft_model(self.policy_model.char_level_decoder.base, lora_cfg)

    def forward(self, *args, **kwargs):
        input_ids = kwargs.get("input_ids")
        attention_mask = kwargs.get("attention_mask")
        if input_ids is None and args:
            input_ids = args[0]
        if input_ids is None:
            raise ValueError("NotaGenHFAdapter.forward requires input_ids")
        logits = self.wrapper.forward_logits(input_ids=input_ids, attention_mask=attention_mask)
        return CausalLMOutput(logits=logits)

    def add_model_tags(self, tags):
        self._trl_tags = list(tags)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self._gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self._gradient_checkpointing = False

    @property
    def is_gradient_checkpointing(self):
        return bool(self._gradient_checkpointing)

    def generate_batch(
        self,
        prompt_ids: list[str] | list[list[int]],
        generation_config: GenerationConfig | None = None,
        continuous_batching_config: Any | None = None,
        progress_bar: bool = False,
        **kwargs,
    ) -> dict[int, NotaGenGeneratedOutput]:
        if not prompt_ids:
            return {}
        if not isinstance(prompt_ids[0], str):
            raise NotImplementedError(
                "NotaGenHFAdapter.generate_batch expects raw prompt strings, not token IDs."
            )

        cfg = generation_config or self.generation_config
        results = self.wrapper.generate_batch(
            prompt_ids,
            temperature=float(getattr(cfg, "temperature", 1.0) or 1.0),
            top_k=int(getattr(cfg, "top_k", 8) or 8),
            top_p=float(getattr(cfg, "top_p", 0.95) or 0.95),
            target_stream_lines=int(kwargs.pop("target_stream_lines", 8)),
            max_chars=int(kwargs.pop("max_chars", 16000)),
            timeout_s=int(kwargs.pop("timeout_s", 45)),
            seed=int(kwargs.pop("seed", 0)),
        )
        return {
            i: NotaGenGeneratedOutput(generated_tokens=self.wrapper.encode_text_ids(result.completion))
            for i, result in enumerate(results)
        }

    def score_completions(self, prompt_completion_pairs: list[tuple[str, str]]) -> list[list[float]]:
        return self.wrapper.score_batch(prompt_completion_pairs)

    def to(self, *args, **kwargs):
        # Keep torch.Module semantics but note that the real device move already happened in the wrapper.
        result = super().to(*args, **kwargs)
        if args and isinstance(args[0], torch.device):
            self.device_ref = args[0]
        elif "device" in kwargs and kwargs["device"] is not None:
            self.device_ref = kwargs["device"]
        return result


def build_trl_ready_notagen(
    weights_path: str,
    *,
    precision: str = "bf16",
    device: torch.device | None = None,
    use_lora: bool = True,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
):
    model = NotaGenHFAdapter(
        weights_path=weights_path,
        precision=precision,
        device=device,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    processing = NotaGenProcessingStub()
    return SimpleNamespace(model=model, processing_class=processing)
