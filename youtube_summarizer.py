from urllib.parse import urlparse, parse_qs
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import time

import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq


# =====================================================
# STREAMLIT CONFIG
# =====================================================

st.set_page_config(
    page_title="YouTube Video Summarizer",
    page_icon="🎥",
    layout="centered"
)

st.title("🎥 YouTube Video Summarizer")
st.write("Paste a YouTube video link and get an AI summary instantly.")


# =====================================================
# THRESHOLDS
# =====================================================

SHORT_THRESHOLD  = 20_000
MEDIUM_THRESHOLD = 50_000
CHUNK_SIZE       = 8_000
CHUNK_OVERLAP    = 100


# =====================================================
# CACHE LLM
# llama-3.1-8b-instant → fastest + highest free limits
# =====================================================

@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=st.secrets["GROQ_API_KEY"],
        temperature=0,
    )


# =====================================================
# EXTRACT VIDEO ID
# =====================================================

def extract_video_id(youtube_url: str):
    parsed = urlparse(youtube_url)

    if parsed.hostname == "youtu.be":
        return parsed.path[1:]

    if parsed.hostname in ("www.youtube.com", "youtube.com"):

        if "/shorts/" in parsed.path:
            return parsed.path.split("/shorts/")[1].split("?")[0]

        return parse_qs(parsed.query).get("v", [None])[0]

    return None


# =====================================================
# FETCH TRANSCRIPT — cached per video ID
# =====================================================

@st.cache_data(show_spinner=False)
def get_video_transcript(video_id: str) -> str:
    ytt_api = YouTubeTranscriptApi()
    fetched = ytt_api.fetch(video_id)
    return " ".join(snippet.text for snippet in fetched.snippets)


# =====================================================
# COMPRESS TRANSCRIPT
# =====================================================

def compress_transcript(text: str) -> str:
    fillers = (
        r'\b(um+|uh+|like|you know|i mean|basically|literally|'
        r'actually|so|right|okay|ok|yeah|alright|anyway|'
        r'kind of|sort of|you see|you know what i mean)\b'
    )
    text = re.sub(fillers, "", text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# =====================================================
# SMART SPLIT
# =====================================================

def split_transcript(text: str) -> list:
    if len(text) <= SHORT_THRESHOLD:
        return [text]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_text(text)


# =====================================================
# SUMMARISE A SINGLE CHUNK
# time.sleep → prevents hitting Groq rate limit
# =====================================================

def summarise_chunk(chunk: str, llm, idx: int, total: int) -> str:
    time.sleep(5)   # prevents 429 rate limit errors
    prompt = (
        f"Section {idx + 1} of {total}. "
        "Extract ONLY the key points as bullet points. "
        "Be concise — no intro sentence, no conclusion, just the bullets.\n\n"
        f"{chunk}"
    )
    return llm.invoke([HumanMessage(content=prompt)]).content


# =====================================================
# USER INPUT
# =====================================================

youtube_url = st.text_input("Enter YouTube Video URL")


# =====================================================
# BUTTON ACTION
# =====================================================

if st.button("Generate Summary"):

    if not youtube_url.strip():
        st.warning("Please enter a YouTube URL")
        st.stop()

    try:

        # Step 1: Video ID
        video_id = extract_video_id(youtube_url)
        if not video_id:
            st.error("Invalid YouTube URL — please check and try again.")
            st.stop()

        # Step 2: Transcript
        with st.spinner("Fetching transcript…"):
            raw_transcript = get_video_transcript(video_id)

        # Step 3: Compress
        transcript = compress_transcript(raw_transcript)

        transcript = transcript[:40_000] 

        # Step 4: Split
        chunks = split_transcript(transcript)

        llm = load_llm()
        progress = st.progress(0, text="Starting…")

        # Step 5A: SHORT VIDEO — single direct call
        if len(chunks) == 1:
            progress.progress(30, text="Summarising…")
            final_prompt = (
                "Create a clear, structured summary of this YouTube video.\n"
                "Format:\n"
                "- Short intro paragraph (2-3 sentences)\n"
                "- Key Takeaways as bullet points\n\n"
                f"Transcript:\n{chunks[0]}"
            )
            st.subheader("📌 Video Summary")
            box = st.empty()
            out = ""
            for token in llm.stream([HumanMessage(content=final_prompt)]):
                out += token.content
                box.markdown(out + "▌")
            box.markdown(out)
            progress.progress(100, text="Done ✅")

        # Step 5B: LONGER VIDEO — parallel map → single reduce
        else:
            chunk_summaries = [""] * len(chunks)
            completed = 0

            with ThreadPoolExecutor(max_workers=1) as executor:
                futures = {
                    executor.submit(summarise_chunk, chunk, llm, idx, len(chunks)): idx
                    for idx, chunk in enumerate(chunks)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    chunk_summaries[idx] = future.result()
                    completed += 1
                    pct = int((completed / len(chunks)) * 75)
                    progress.progress(
                        pct,
                        text=f"Processed {completed}/{len(chunks)} sections…"
                    )

            # REDUCE: single final call
            progress.progress(80, text="Merging into final summary…")
            combined = "\n\n".join(chunk_summaries)
            final_prompt = (
                "You have bullet-point summaries of different sections of a YouTube video.\n"
                "Merge them into ONE clean final summary:\n"
                "- Short intro paragraph (2-3 sentences)\n"
                "- Organised bullet points grouped by theme\n"
                "- Remove all repetition\n\n"
                f"{combined}"
            )

            st.subheader("📌 Video Summary")
            box = st.empty()
            out = ""
            for token in llm.stream([HumanMessage(content=final_prompt)]):
                out += token.content
                box.markdown(out + "▌")
            box.markdown(out)
            progress.progress(100, text="Done ✅")

    except Exception as e:
        st.error(f"Error: {e}")
