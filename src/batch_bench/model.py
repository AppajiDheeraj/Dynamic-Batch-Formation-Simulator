from __future__ import annotations

from collections.abc import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .schedulers import ActiveRequest, RequestSpec


class QwenEngine:
    def __init__(self, model_name: str) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required. Run this benchmark on an NVIDIA GPU.")
        self.device = torch.device("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        ).to(self.device)
        self.model.eval()
        eos = self.model.generation_config.eos_token_id
        if eos is None:
            eos = self.tokenizer.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()

    def activate(self, spec: RequestSpec, started_ms: float) -> ActiveRequest:
        input_ids = self.tokenizer.encode(
            spec.prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=128,
        )
        if not input_ids:
            raise ValueError(f"request {spec.id} produced an empty prompt")
        return ActiveRequest(
            spec=spec,
            input_ids=input_ids,
            generated_ids=[],
            prompt_tokens=len(input_ids),
            started_ms=started_ms,
        )

    @torch.inference_mode()
    def step(self, active: Sequence[ActiveRequest]) -> None:
        sequences = [request.input_ids + request.generated_ids for request in active]
        width = max(len(sequence) for sequence in sequences)
        pad_id = self.tokenizer.pad_token_id
        padded = [[pad_id] * (width - len(sequence)) + sequence for sequence in sequences]
        masks = [[0] * (width - len(sequence)) + [1] * len(sequence) for sequence in sequences]
        input_ids = torch.tensor(padded, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(masks, dtype=torch.long, device=self.device)
        position_ids = attention_mask.cumsum(dim=1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)

        # shortcut: full sequences are recomputed, add a batched KV cache when this becomes the measured bottleneck.
        logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        ).logits[:, -1, :]
        next_ids = torch.argmax(logits, dim=-1).tolist()
        for request, token_id in zip(active, next_ids):
            request.generated_ids.append(token_id)
            request.eos_reached = token_id in self.eos_ids
        torch.cuda.synchronize()

    def decode(self, request: ActiveRequest) -> str:
        return self.tokenizer.decode(request.generated_ids, skip_special_tokens=True)

    def warm_up(self) -> None:
        spec = RequestSpec("warmup", "Hello", 0, 2)
        request = self.activate(spec, 0)
        self.step([request])
        self.step([request])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    def peak_memory_bytes(self) -> int:
        return torch.cuda.max_memory_allocated()
