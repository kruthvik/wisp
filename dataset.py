#!/usr/bin/env python3
"""
Build a local forced-alignment / transcript-correction test set from LibriSpeech.

What it creates:
    alignment_testset/
        audio/
            0000_<id>.flac
            0001_<id>.flac
            ...
        transcripts/
            0000_<id>.txt
            0001_<id>.txt
            ...
        manifest.jsonl
        README.txt

Each manifest line contains:
    {
      "id": "...",
      "audio": "audio/....flac",
      "transcript": "transcripts/....txt",
      "text": "...",
      "speaker_id": ...,
      "chapter_id": ...
    }

Install:
    pip install -U datasets huggingface_hub

Usage:
    python make_alignment_testset.py
    python make_alignment_testset.py --count 250
    python make_alignment_testset.py --count 100 --config other
    python make_alignment_testset.py --count 100 --split test

Notes:
- Uses LibriSpeech, a standard ASR corpus with reference transcripts.
- Streaming is used so you don't download the entire dataset.
- Audio is kept in its original FLAC form when possible.
"""

import argparse
import json
import shutil
from pathlib import Path

from datasets import Audio, load_dataset


DATASET_NAME = "openslr/librispeech_asr"


def safe_name(value: str) -> str:
    return "".join(
        c if c.isalnum() or c in "-_" else "_"
        for c in str(value)
    )


def save_audio(audio_obj, destination: Path) -> None:
    """
    Saves an Audio(decode=False) example.

    Modern Hugging Face Datasets normally returns:
        {"bytes": b"...", "path": "..."}
    for decode=False. This function handles both bytes and path cases.
    """
    if audio_obj is None:
        raise RuntimeError("Dataset example has no audio data.")

    # Common decode=False dictionary representation
    if isinstance(audio_obj, dict):
        audio_bytes = audio_obj.get("bytes")
        audio_path = audio_obj.get("path")

        if audio_bytes is not None:
            destination.write_bytes(audio_bytes)
            return

        if audio_path:
            source = Path(audio_path)
            if source.exists():
                shutil.copyfile(source, destination)
                return

    # Some versions may expose an object with .path
    audio_path = getattr(audio_obj, "path", None)
    if audio_path:
        source = Path(audio_path)
        if source.exists():
            shutil.copyfile(source, destination)
            return

    raise RuntimeError(
        "Could not extract the audio bytes/path. "
        "Try upgrading datasets: pip install -U datasets"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of audio/transcript pairs to save (default: 100)",
    )
    parser.add_argument(
        "--config",
        choices=["clean", "other"],
        default="clean",
        help="'clean' is easier audio; 'other' is harder (default: clean)",
    )
    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="validation",
        help="Dataset split to sample from (default: validation)",
    )
    parser.add_argument(
        "--output",
        default="alignment_testset",
        help="Output directory (default: alignment_testset)",
    )
    args = parser.parse_args()

    if args.count <= 0:
        raise SystemExit("--count must be greater than 0")

    root = Path(args.output)
    audio_dir = root / "audio"
    transcript_dir = root / "transcripts"

    audio_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Loading {DATASET_NAME} "
        f"config={args.config!r}, split={args.split!r} in streaming mode..."
    )

    dataset = load_dataset(
        DATASET_NAME,
        args.config,
        split=args.split,
        streaming=True,
    )

    # We only need original encoded audio bytes/path.
    # Avoid decoding thousands of samples unnecessarily.
    dataset = dataset.cast_column("audio", Audio(decode=False))

    manifest_path = root / "manifest.jsonl"

    saved = 0

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for example in dataset:
            if saved >= args.count:
                break

            sample_id = safe_name(example["id"])
            stem = f"{saved:04d}_{sample_id}"

            audio_path = audio_dir / f"{stem}.flac"
            transcript_path = transcript_dir / f"{stem}.txt"

            text = str(example["text"]).strip()

            try:
                save_audio(example["audio"], audio_path)
            except Exception as exc:
                print(f"[skip] {sample_id}: {exc}")
                continue

            transcript_path.write_text(text + "\n", encoding="utf-8")

            record = {
                "index": saved,
                "id": str(example["id"]),
                "audio": str(audio_path.relative_to(root)),
                "transcript": str(transcript_path.relative_to(root)),
                "text": text,
                "speaker_id": example.get("speaker_id"),
                "chapter_id": example.get("chapter_id"),
                "config": args.config,
                "split": args.split,
            }

            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")

            saved += 1
            print(f"[{saved:4d}/{args.count}] {sample_id}")

    readme = f"""Alignment Test Set

Source:
    {DATASET_NAME}

Configuration:
    {args.config}

Split:
    {args.split}

Samples:
    {saved}

Structure:
    audio/          Original FLAC recordings
    transcripts/    Reference transcript for each recording
    manifest.jsonl  Machine-readable mapping between audio and transcript

Suggested first tests:

    1. Easy baseline:
       python make_alignment_testset.py --count 100 --config clean

    2. Larger clean test:
       python make_alignment_testset.py --count 500 --config clean

    3. Harder speech:
       python make_alignment_testset.py --count 100 --config other

Important:
    These transcripts should be treated as the reference text for testing.
    Your program should receive the audio plus its corresponding .txt file.

For alignment testing, don't only measure whether the final words equal the
reference. Also record how many reference words got:
    - an exact Whisper-derived timestamp
    - an estimated/interpolated timestamp
    - no usable timestamp
"""
    (root / "README.txt").write_text(readme, encoding="utf-8")

    print()
    print(f"Done. Saved {saved} samples to: {root.resolve()}")
    print(f"Manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
