from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class VideoCaption:
    video_id: str
    caption: str


class MSRVTTLoader:
    """Загрузчик аннотаций MSR-VTT. По-хорошему надо сделать базовый лроадер и от него наследоваться

    Должен быть файл вида train_val_videodatainfo.json со схемой
    {"videos": [...], "sentences": [{"video_id", "caption", "sen_id"}, ...]}.
    """

    def __init__(self, annotations_path: str | Path, split: str | None = None):
        self.annotations_path = Path(annotations_path)
        self.split = split  # "train" | "validate" | "test" | None

    def _video_ids_for_split(self, payload: dict) -> set[str] | None:
        if self.split is None:
            return None
        return {v["video_id"] for v in payload["videos"] if v["split"] == self.split}

    def __iter__(self) -> Iterator[VideoCaption]:
        with self.annotations_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        allowed = self._video_ids_for_split(payload)
        for s in payload["sentences"]:
            if allowed is not None and s["video_id"] not in allowed:
                continue
            caption = s["caption"].strip()
            if not caption:
                continue
            yield VideoCaption(video_id=s["video_id"], caption=caption)
