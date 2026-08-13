# wisp

`wisp` aligns spoken audio with a reference transcript and generates word-level timestamps in SRT format.

## What it does

- Transcribes audio with [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- Aligns transcription words to a provided transcript
- Fills missing timestamps for unmatched words using interpolation
- Writes an output `.srt` file with per-word timing
- Provides a Streamlit UI for upload, transcription, subtitle preview, and SRT download

## Requirements

- Python 3.10+
- Dependencies from `requirements.txt`

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run main.py
```

Then in the UI:
1. Upload an audio file (`wav`, `mp3`, `m4a`, `flac`, `webm`)
2. Upload a transcript (`.txt`)
3. Choose model/device/compute settings
4. Click **Transcribe**
5. Download the generated `output.srt`

## Programmatic usage

```python
from wisp import transcribe_audio

aligned_words = transcribe_audio(
    audio_path="audio.wav",
    transcript_path="transcript.txt",
    output_path="output.srt",
    model_size="tiny",
    device="cpu",
    compute_type="int8",
    language="en",
)
```

Each item in `aligned_words` includes:
- `text`
- `start`
- `end`
- `matched` (whether it matched directly to Whisper output)

## Repository structure

- `/home/runner/work/wisp/wisp/main.py` — Streamlit interface
- `/home/runner/work/wisp/wisp/wisp.py` — alignment and SRT generation logic
- `/home/runner/work/wisp/wisp/dataset.py` — helper script to build an alignment test set
