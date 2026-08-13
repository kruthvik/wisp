import base64
import json
from html import escape
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from wisp import AlignedWord, transcribe_audio


def show_subtitled_audio(
    audio_bytes: bytes, mime_type: str, aligned_words: list[AlignedWord]
) -> None:
    """Render audio with a visible caption panel synchronized to word timings."""
    encoded_audio = base64.b64encode(audio_bytes).decode("ascii")
    audio_source = f"data:{mime_type};base64,{encoded_audio}"
    timed_words = [
        {"text": word["text"], "start": word["start"], "end": word["end"]}
        for word in aligned_words
        if word["start"] is not None and word["end"] is not None
    ]
    words_json = json.dumps(timed_words).replace("</", "<\\/")

    components.html(
        f'''<div style="font-family: sans-serif;">
            <audio id="wisp-audio" controls style="width: 100%;" preload="metadata">
                <source src="{escape(audio_source, quote=True)}">
                Your browser does not support audio playback.
            </audio>
            <div id="wisp-caption" aria-live="polite" style="
                min-height: 3.5rem; margin-top: 0.75rem; padding: 1rem;
                border-radius: 0.5rem; background: #1f2937; color: white;
                font-size: 1.5rem; font-weight: 600; text-align: center;">
                press play to view subtitles
            </div>
        </div>
        <script>
            const audio = document.getElementById("wisp-audio");
            const caption = document.getElementById("wisp-caption");
            const words = {words_json};

            function updateCaption() {{
                const time = audio.currentTime;
                const activeIndex = words.findIndex(
                    word => time >= word.start && time <= word.end
                );
                if (activeIndex === -1) return;

                const start = Math.max(0, activeIndex - 3);
                const end = Math.min(words.length, activeIndex + 4);
                caption.textContent = words
                    .slice(start, end)
                    .map((word, index) => index === activeIndex - start
                        ? "[" + word.text + "]"
                        : word.text)
                    .join(" ");
            }}

            audio.addEventListener("timeupdate", updateCaption);
            audio.addEventListener("seeked", updateCaption);
            audio.addEventListener("ended", () => {{
                caption.textContent = "Playback finished";
            }});
        </script>''',
        height=150,
    )


def main():
    st.title("wisp")
    st.write("aligns spoken audio with reference transcripts")

    audio_path = st.file_uploader("upload audio file", type=["wav", "mp3", "m4a", "flac", "webm"])
    transcript_path = st.file_uploader("upload transcript file", type=["txt"])
    output_path = st.text_input("output SRT file path", value="output.srt")
    model_size = st.selectbox("model size", ["tiny", "base", "small", "medium", "large"])
    device = st.selectbox("device", ["cpu", "cuda"])
    compute_type = st.selectbox("compute type", ["int8", "float16", "float32"])
    language = st.text_input("language (optional)", value="en")

    if st.button("Transcribe"):
        if audio_path and transcript_path:
            with st.spinner("Transcribing and aligning audio..."):
                aligned_words = transcribe_audio(
                    audio_path=audio_path.getvalue(),
                    transcript_path=transcript_path.getvalue(),
                    output_path=output_path,
                    model_size=model_size,
                    device=device,
                    compute_type=compute_type,
                    language=language or None,
                )

            st.session_state["transcription_result"] = {
                "audio_bytes": audio_path.getvalue(),
                "audio_mime_type": audio_path.type or "audio/mpeg",
                "srt_bytes": Path(output_path).read_bytes(),
                "aligned_words": aligned_words,
            }
            st.success("Transcription complete!")
        else:
            st.error("Please upload both an audio file and a transcript file.")

    result = st.session_state.get("transcription_result")
    if result:
        st.download_button(
            "Download generated SRT",
            data=result["srt_bytes"],
            file_name="output.srt",
            mime="application/x-subrip",
        )
        st.subheader("Audio with subtitles")
        show_subtitled_audio(
            result["audio_bytes"],
            result["audio_mime_type"],
            result["aligned_words"],
        )

if __name__ == "__main__":
    main()

# audio_path
# transcript_path
# output_path
# model_size
# device
# compute_type
# language
# word_timestamps
# model
