from urllib.parse import urlparse, parse_qs
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import time
import requests as req_lib

import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from fpdf import FPDF


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
# CONSTANTS
# =====================================================

SHORT_THRESHOLD  = 20_000
MEDIUM_THRESHOLD = 50_000
CHUNK_SIZE       = 4_000
CHUNK_OVERLAP    = 100


# =====================================================
# CACHE LLM
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
# FETCH TRANSCRIPT — via Supadata API
# =====================================================

@st.cache_data(show_spinner=False)
def get_video_transcript(video_id: str) -> str:
    response = req_lib.get(
        "https://api.supadata.ai/v1/youtube/transcript",
        params={"videoId": video_id, "lang": "en"},
        headers={"x-api-key": st.secrets["SUPADATA_API_KEY"]}
    )
    data = response.json()
    if "content" not in data:
        raise Exception("Transcript not available for this video.")
    return " ".join([item["text"] for item in data["content"]])


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
# SUMMARISE A SINGLE CHUNK (map step)
# =====================================================

def summarise_chunk(chunk: str, llm, idx: int, total: int) -> str:
    time.sleep(5)
    prompt = (
        f"Section {idx + 1} of {total}. "
        "Extract ONLY the key points as brief bullet points. "
        "No intro, no conclusion, just bullets.\n\n"
        f"{chunk}"
    )
    return llm.invoke([HumanMessage(content=prompt)]).content


# =====================================================
# BUILD FINAL PROMPT
# Based on user-selected style, length, language
# =====================================================

def build_final_prompt(content: str, style: str, length: str, language: str) -> str:
    style_map = {
        "Bullet Points": "Format as clear bullet points with a brief 1-2 sentence intro paragraph.",
        "Paragraph":     "Format as well-written flowing paragraphs.",
        "Key Takeaways": "Format as numbered key takeaways (1. 2. 3. etc.) with a brief intro.",
    }
    length_map = {
        "Short":    "Be very concise — cover only the 3-5 most important points.",
        "Medium":   "Cover all main topics in moderate detail.",
        "Detailed": "Be comprehensive — cover all topics, subtopics, and examples mentioned.",
    }
    lang_note = f"\n\nIMPORTANT: Write the entire summary in {language}." if language != "English" else ""

    return f"""Create a structured summary of this YouTube video.
{style_map[style]}
{length_map[length]}{lang_note}

Content:
{content}"""


# =====================================================
# EXTRACT KEYWORDS
# =====================================================

def extract_keywords(transcript: str, llm) -> list:
    time.sleep(5)   # ← add this line at the top
    prompt = (
        "Extract exactly 8 important keywords or key phrases from this transcript. "
        "Return ONLY a comma-separated list, nothing else.\n\n"
        f"{transcript[:4000]}"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    keywords = [k.strip() for k in response.content.split(',')]
    return [k for k in keywords if k][:8]


# =====================================================
# GENERATE PDF
# =====================================================

def create_pdf(video_id: str, summary: str, keywords: list, style: str, length: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font('Helvetica', 'B', 18)
    pdf.cell(0, 12, 'YouTube Video Summary', ln=True, align='C')
    pdf.ln(2)

    # Divider
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)

    # Metadata
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 5, f'Source: https://youtube.com/watch?v={video_id}', ln=True)
    pdf.cell(0, 5, f'Style: {style}   |   Length: {length}', ln=True)
    pdf.ln(5)

    # Keywords
    if keywords:
        pdf.set_font('Helvetica', 'B', 12)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 8, 'Keywords:', ln=True)
        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(60, 60, 60)
        pdf.multi_cell(0, 6, ', '.join(keywords))
        pdf.ln(4)

    # Summary
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, 'Summary:', ln=True)
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(30, 30, 30)

    # Strip markdown for PDF
    clean = re.sub(r'\*\*(.+?)\*\*', r'\1', summary)
    clean = re.sub(r'\*(.+?)\*', r'\1', clean)
    clean = re.sub(r'#{1,6}\s', '', clean)
    pdf.multi_cell(0, 6, clean)

    return bytes(pdf.output())


# =====================================================
# USER OPTIONS
# =====================================================

col1, col2, col3 = st.columns(3)
with col1:
    style = st.selectbox("📝 Summary style", ["Bullet Points", "Paragraph", "Key Takeaways"])
with col2:
    length = st.selectbox("📏 Summary length", ["Short", "Medium", "Detailed"])
with col3:
    language = st.selectbox("🌐 Language", [
        "English", "Hindi", "Telugu", "Tamil",
        "Spanish", "French", "German", "Arabic"
    ])


# =====================================================
# URL INPUT + THUMBNAIL PREVIEW
# =====================================================

youtube_url = st.text_input("Enter YouTube Video URL")

# Show thumbnail preview as soon as URL is pasted
if youtube_url.strip():
    preview_id = extract_video_id(youtube_url.strip())
    if preview_id:
        col_t, col_i = st.columns([1, 2])
        with col_t:
            st.image(
                f"https://img.youtube.com/vi/{preview_id}/mqdefault.jpg",
                use_container_width=True
            )
        with col_i:
            st.success("✅ Valid YouTube URL detected")
            st.caption(f"Video ID: `{preview_id}`")
            st.caption(f"Style: {style}  ·  Length: {length}  ·  Language: {language}")


# =====================================================
# BUTTON ACTION
# =====================================================

if st.button("🚀 Generate Summary", type="primary", use_container_width=True):

    if not youtube_url.strip():
        st.warning("Please enter a YouTube URL")
        st.stop()

    out = ""
    keywords = []

    try:

        # Step 1: Video ID
        video_id = extract_video_id(youtube_url)
        if not video_id:
            st.error("Invalid YouTube URL — please check and try again.")
            st.stop()

        # Step 2: Transcript
        with st.spinner("Fetching transcript…"):
            raw_transcript = get_video_transcript(video_id)

        # Step 3: Compress + trim
        transcript = compress_transcript(raw_transcript)
        transcript = transcript[:40_000]

        # Step 4: Split
        chunks = split_transcript(transcript)

        llm = load_llm()
        progress = st.progress(0, text="Starting…")

        # ── SHORT VIDEO — single direct call ──────────────────────────
        if len(chunks) == 1:
            progress.progress(20, text="Extracting keywords…")
            keywords = extract_keywords(transcript, llm)

            progress.progress(50, text="Summarising…")
            final_prompt = build_final_prompt(chunks[0], style, length, language)

            # Keywords display
            st.subheader("🔑 Keywords")
            kw_html = " ".join([
                f'<span style="background:#4f46e5;color:white;padding:3px 12px;'
                f'border-radius:20px;font-size:12px;margin:2px;display:inline-block;">{k}</span>'
                for k in keywords
            ])
            st.markdown(kw_html, unsafe_allow_html=True)
            st.divider()

            # Summary
            st.subheader("📌 Video Summary")
            box = st.empty()
            for token in llm.stream([HumanMessage(content=final_prompt)]):
                out += token.content
                box.markdown(out + "▌")
            box.markdown(out)
            progress.progress(100, text="Done ✅")

        # ── LONGER VIDEO — parallel map → single reduce ───────────────
        else:
            chunk_summaries = [""] * len(chunks)
            completed = 0

            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = {
                    executor.submit(summarise_chunk, chunk, llm, idx, len(chunks)): idx
                    for idx, chunk in enumerate(chunks)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    chunk_summaries[idx] = future.result()
                    completed += 1
                    pct = int((completed / len(chunks)) * 60)
                    progress.progress(pct, text=f"Processed {completed}/{len(chunks)} sections…")

            # Keywords
            progress.progress(65, text="Extracting keywords…")
            keywords = extract_keywords(transcript, llm)

            # Reduce
            progress.progress(75, text="Merging into final summary…")
            combined = "\n\n".join(chunk_summaries)
            final_prompt = build_final_prompt(combined, style, length, language)

            # Keywords display
            st.subheader("🔑 Keywords")
            kw_html = " ".join([
                f'<span style="background:#4f46e5;color:white;padding:3px 12px;'
                f'border-radius:20px;font-size:12px;margin:2px;display:inline-block;">{k}</span>'
                for k in keywords
            ])
            st.markdown(kw_html, unsafe_allow_html=True)
            st.divider()

            # Summary
            st.subheader("📌 Video Summary")
            box = st.empty()
            for token in llm.stream([HumanMessage(content=final_prompt)]):
                out += token.content
                box.markdown(out + "▌")
            box.markdown(out)
            progress.progress(100, text="Done ✅")

        # ── Post-summary actions ──────────────────────────────────────
        st.divider()
        col_txt, col_pdf = st.columns(2)

        with col_txt:
            st.download_button(
                label="📋 Download as Text",
                data=out,
                file_name="summary.txt",
                mime="text/plain",
                use_container_width=True,
            )

        with col_pdf:
            pdf_bytes = create_pdf(video_id, out, keywords, style, length)
            st.download_button(
                label="📄 Download as PDF",
                data=pdf_bytes,
                file_name="summary.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        # Copy text expander
        with st.expander("📋 Copy summary text"):
            st.code(out, language=None)

    except Exception as e:
        st.error(f"Error: {e}")
