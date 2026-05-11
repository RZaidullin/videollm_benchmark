from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


@dataclass(frozen=True)
class GeneratedQuestion:
    video_id: str
    caption: str
    question: str
    answer: str  # span из caption, выделенный <hl>...<hl>


class T5QuestionGenerator:
    """Генерация вопросов из описания видео.

    Использует T5, дообученную на SQuAD v1.1.
    Поддерживает два сценария:
      * answer-aware: вопросы строятся вокруг заранее выбранных span-ов;
      * e2e: модель сама выделяет answer-кандидатов (использует
        valhalla/t5-base-e2e-qg).
    """

    def __init__(
        self,
        model_name: str = "valhalla/t5-base-qg-hl",
        device: str | None = None,
        max_input_length: int = 512,
        max_output_length: int = 64,
        num_beams: int = 4,
        num_return_sequences: int = 1,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.num_beams = num_beams
        self.num_return_sequences = num_return_sequences

    @staticmethod
    def _extract_answer_spans(caption: str) -> list[str]:
        from string import punctuation
        stop = {"the", "a", "an", "is", "are", "was", "were", "and", "or",
                "of", "in", "on", "at", "to", "for", "with", "by"}
        tokens = [t.strip(punctuation) for t in caption.split()]
        return [t for t in tokens if len(t) > 3 and t.lower() not in stop]

    def _build_input(self, caption: str, answer: str) -> str:
        # Будет так "generate question: <text-with-<hl>answer<hl>-marker> </s>"
        marked = caption.replace(answer, f"<hl> {answer} <hl>", 1)
        return f"generate question: {marked} </s>"

    @torch.inference_mode()
    def generate(self, video_id: str, caption: str) -> list[GeneratedQuestion]:
        spans = self._extract_answer_spans(caption)
        if not spans:
            return []

        prompts = [self._build_input(caption, a) for a in spans]
        enc = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_input_length,
            return_tensors="pt",
        ).to(self.device)

        out = self.model.generate(
            **enc,
            max_length=self.max_output_length,
            num_beams=self.num_beams,
            num_return_sequences=self.num_return_sequences,
            early_stopping=True,
        )
        decoded = self.tokenizer.batch_decode(out, skip_special_tokens=True)

        # Если num_return_sequences > 1, decoded длиннее prompts — разворачиваем по answer.
        questions: list[GeneratedQuestion] = []
        per_prompt = self.num_return_sequences
        for i, answer in enumerate(spans):
            for j in range(per_prompt):
                q = decoded[i * per_prompt + j].strip()
                if q:
                    questions.append(GeneratedQuestion(
                        video_id=video_id, caption=caption, question=q, answer=answer,
                    ))
        return self._deduplicate(questions)

    @staticmethod
    def _deduplicate(items: list[GeneratedQuestion]) -> list[GeneratedQuestion]:
        seen: set[str] = set()
        result = []
        for it in items:
            key = it.question.lower().rstrip("?.! ")
            if key in seen:
                continue
            seen.add(key)
            result.append(it)
        return result
