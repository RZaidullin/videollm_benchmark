from __future__ import annotations
import logging
from pathlib import Path
from typing import Iterator

from tqdm import tqdm

from src.dataset_loader import MSRVTTLoader, VideoCaption
from src.question_generator import T5QuestionGenerator, GeneratedQuestion
from src.question_classifier import (
    BartZeroShotQuestionClassifier,
    ClassifiedQuestion,
)
from src.dataset_writer import JsonlDatasetWriter

log = logging.getLogger(__name__)


def run_generation(
    annotations_path: str,
    output_path: str,
    split: str | None = "validate",
    qg_model: str = "valhalla/t5-base-qg-hl",
    nli_model: str = "facebook/bart-large-mnli",
    classifier_threshold: float = 0.55,
    max_videos: int | None = None,
) -> int:
    loader = MSRVTTLoader(annotations_path, split=split)
    qg = T5QuestionGenerator(model_name=qg_model)
    clf = BartZeroShotQuestionClassifier(
        model_name=nli_model,
        confidence_threshold=classifier_threshold,
    )
    writer = JsonlDatasetWriter(output_path)

    def stream() -> Iterator[ClassifiedQuestion]:
        seen_videos: set[str] = set()
        for vc in tqdm(loader, desc="captions"):
            if max_videos and len(seen_videos) >= max_videos and vc.video_id not in seen_videos:
                continue
            seen_videos.add(vc.video_id)

            questions = qg.generate(vc.video_id, vc.caption)
            if not questions:
                continue

            qtypes = clf.classify_batch([q.question for q in questions])
            for q, (qtype, score) in zip(questions, qtypes):
                if score < classifier_threshold:
                    continue  # пропускаем неуверенно классифицированные
                yield ClassifiedQuestion(
                    video_id=q.video_id,
                    caption=q.caption,
                    question=q.question,
                    answer=q.answer,
                    qtype=qtype,
                    score=score,
                )

    n = writer.write(stream())
    log.info("Generated dataset: %d items → %s", n, output_path)
    return n
