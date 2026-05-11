from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable

from .question_classifier import ClassifiedQuestion


class JsonlDatasetWriter:
    """Будем хранить троки в json - надо наверно добавить в текст - не забыть"""
    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, items: Iterable[ClassifiedQuestion]) -> int:
        n = 0
        with self.output_path.open("w", encoding="utf-8") as f:
            for it in items:
                record = {
                    "video_id": it.video_id,
                    "caption": it.caption,
                    "question": it.question,
                    "reference_answer": it.answer,
                    "question_type": it.qtype,
                    "classifier_score": round(it.score, 4),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n += 1
        return n
