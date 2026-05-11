from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

import torch
from transformers import pipeline

QuestionType = Literal["temporal", "semantic"]


@dataclass(frozen=True)
class ClassifiedQuestion:
    video_id: str
    caption: str
    question: str
    answer: str
    qtype: QuestionType
    score: float


TEMPORAL_KEYWORDS: frozenset[str] = frozenset({
    "when", "before", "after", "while", "during", "until",
    "then", "first", "next", "last", "finally",
    "how long", "how often", "in what order",
})


class BartZeroShotQuestionClassifier:
    """Классификация вопросов на 'temporal' / 'semantic' (временаные/семантические)
    через BART-large-MNLI."""

    HYPOTHESIS_TEMPLATE = "This question is {}."
    LABELS = [
        ("temporal",
         "temporal — about time, order, duration, or sequence of events "
         "(keywords: when, before, after, during, while, until, how long, "
         "how often, first, then, finally)"),
        ("semantic",
         "semantic — about objects, attributes, actions, or relations in the scene "
         "(keywords: what, who, where, which, how many, what color, what kind, "
         "object, action)"),
    ]

    def __init__(
        self,
        model_name: str = "facebook/bart-large-mnli",
        device: int | None = None,
        confidence_threshold: float = 0.55,
        use_keyword_prefilter: bool = True,
    ):
        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=device,
        )
        self.threshold = confidence_threshold
        self.use_prefilter = use_keyword_prefilter

        self._label_keys = [k for k, _ in self.LABELS]
        self._label_texts = [t for _, t in self.LABELS]
        self._text_to_key = dict(zip(self._label_texts, self._label_keys))

    def _keyword_prefilter(self, question: str) -> QuestionType | None:
        q = question.lower()
        for kw in TEMPORAL_KEYWORDS:
            # для словосочетаний — подстрока; для отдельных слов — по границам
            if " " in kw:
                if kw in q:
                    return "temporal"
            else:
                if f" {kw} " in f" {q} " or q.startswith(kw + " "):
                    return "temporal"
        return None

    def classify_one(self, question: str) -> tuple[QuestionType, float]:
        if self.use_prefilter:
            heuristic = self._keyword_prefilter(question)
            if heuristic is not None:
                return heuristic, 1.0  # эвристическая разметка

        result = self.pipe(
            question,
            candidate_labels=self._label_texts,
            hypothesis_template=self.HYPOTHESIS_TEMPLATE,
            multi_label=False,
        )
        top_label_text = result["labels"][0]
        top_score = float(result["scores"][0])
        qtype = self._text_to_key[top_label_text]
        return qtype, top_score

    def classify_batch(
        self, questions: list[str], batch_size: int = 16,
    ) -> list[tuple[QuestionType, float]]:
        # Pipeline сам управляет батчингом; здесь — простая обёртка.
        out: list[tuple[QuestionType, float]] = []
        for q in questions:
            out.append(self.classify_one(q))
        return out
