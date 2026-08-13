Alignment Test Set

Source:
    openslr/librispeech_asr

Configuration:
    clean

Split:
    validation

Samples:
    100

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
