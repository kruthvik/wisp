"""Backend utilities for aligning a reference transcript to audio timestamps."""

from __future__ import annotations

import re
from io import BytesIO
from difflib import SequenceMatcher
from pathlib import Path
from typing import TypedDict

from faster_whisper import WhisperModel


class WhisperWord(TypedDict):
    text: str
    start: float
    end: float
    prob: float


class AlignedWord(TypedDict):
    text: str
    start: float | None
    end: float | None
    matched: bool


def normalize(word: str) -> str:
    """Normalize a word before matching it with the transcription."""
    return re.sub(r"[^\w']", "", word.lower())


def fill_missing_timestamps(
    aligned: list[AlignedWord], whisper_words: list[WhisperWord]
) -> list[AlignedWord]:
    """Distribute unmatched reference words across neighboring timestamps."""
    index = 0

    while index < len(aligned):
        if aligned[index]["start"] is not None:
            index += 1
            continue

        start_missing = index
        while index < len(aligned) and aligned[index]["start"] is None:
            index += 1
        end_missing = index

        if start_missing > 0:
            region_start = aligned[start_missing - 1]["end"]
            if region_start is None:
                raise RuntimeError("Missing timestamp before an unmatched region.")
        else:
            region_start = whisper_words[0]["start"]

        if end_missing < len(aligned):
            region_end = aligned[end_missing]["start"]
            if region_end is None:
                raise RuntimeError("Missing timestamp after an unmatched region.")
        else:
            region_end = whisper_words[-1]["end"]

        missing_words = aligned[start_missing:end_missing]
        weights = [max(len(word["text"]), 1) for word in missing_words]
        total_weight = sum(weights)
        total_duration = region_end - region_start
        current = region_start

        for word, weight in zip(missing_words, weights):
            duration = total_duration * weight / total_weight
            word["start"] = current
            word["end"] = current + duration
            current += duration

    return aligned


def transcribe_audio(
    audio_path: str | Path | bytes,
    transcript_path: str | Path | bytes,
    output_path: str | Path,
    *,
    model_size: str = "tiny",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str | None = "en",
    model: WhisperModel | None = None,
) -> list[AlignedWord]:
    """Transcribe audio, align it to a reference transcript, and write word-level SRT.

    Args:
        audio_path: Audio file path or audio bytes, such as Streamlit upload data.
        transcript_path: Transcript file path or UTF-8 transcript bytes.
        output_path: Destination for the generated SRT file.
        model_size: Faster Whisper model size or model path to load.
        device: Device passed to Faster Whisper, such as ``"cpu"`` or ``"cuda"``.
        compute_type: Faster Whisper compute type, such as ``"int8"``.
        language: Optional language code. Set to ``None`` for auto-detection.

        model: Optional preloaded Faster Whisper model. When supplied, model loading
            parameters are ignored.

    Returns:
        One dictionary per reference word, including its aligned timestamps and
        whether it was matched directly to the Whisper transcription.

    Raises:
        ValueError: If the transcript has no words or audio yields no word timings.
    """
    output_path = Path(output_path)

    if isinstance(transcript_path, bytes):
        transcript_text = transcript_path.decode("utf-8-sig")
    else:
        transcript_text = Path(transcript_path).read_text(encoding="utf-8")

    reference_words = [normalize(word) for word in transcript_text.split()]
    if not reference_words:
        raise ValueError("The reference transcript contains no words.")

    if model is None:
        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )

    audio_source = BytesIO(audio_path) if isinstance(audio_path, bytes) else str(audio_path)
    segments, _ = model.transcribe(
        audio_source,
        language=language,
        word_timestamps=True,
    )
    whisper_words: list[WhisperWord] = [
        {
            "text": normalize(word.word.strip()),
            "start": float(word.start),
            "end": float(word.end),
            "prob": float(word.probability),
        }
        for segment in segments
        for word in (segment.words or [])
    ]
    if not whisper_words:
        raise ValueError("The audio transcription contains no word timestamps.")

    matcher = SequenceMatcher(
        None,
        [word["text"] for word in whisper_words],
        reference_words,
        autojunk=False,
    )
    aligned: list[AlignedWord] = [
        {"text": word, "start": None, "end": None, "matched": False}
        for word in reference_words
    ]

    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            whisper_index = block.a + offset
            reference_index = block.b + offset
            aligned[reference_index]["start"] = whisper_words[whisper_index]["start"]
            aligned[reference_index]["end"] = whisper_words[whisper_index]["end"]
            aligned[reference_index]["matched"] = True

    aligned = fill_missing_timestamps(aligned, whisper_words)

    with output_path.open("w", encoding="utf-8") as file:
        for index, word in enumerate(aligned):
            _ = file.write(f"{index}\n")
            _ = file.write(f"{word['start']} --> {word['end']}\n")
            _ = file.write(f"{word['text']}\n\n")

    return aligned
