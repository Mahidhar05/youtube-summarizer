from urllib.parse import urlparse, parse_qs
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import time
import os
import requests as req_lib

import streamlit as st
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


# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

.hero {
    text-align: center;
    padding: 2.5rem 1rem 2rem;
}
.hero h1 {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}
.hero p {
    color: #94a3b8;
    font-size: 1rem;
    margin: 0;
}

.glass-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(10px);
}

.section-heading {
    font-size: 1.05rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 0.9rem;
}

.kw-pill {
    display: inline-block;
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    padding: 4px 14px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 500;
    margin: 3px 2px;
}

.info-banner {
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.35);
    border-radius: 10px;
    padding: 0.65rem 1rem;
    color: #a5b4fc;
    font-size: 0.87rem;
    margin-top: 0.5rem;
}

.video-meta {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    color: #94a3b8;
    font-size: 0.85rem;
    margin-top: 0.8rem;
}

div[data-testid="stSelectbox"] > div > div,
div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
}

div[data-testid="stTextInput"] > div > div,
div[data-baseweb="input"] > div {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
}

input, textarea { color: #e2e8f0 !important; }

label, .stSelectbox label, .stTextInput label {
    color: #94a3b8 !important;
    font-size: 0.87rem !important;
    font-weight: 500 !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    color: white !important;
    box-shadow: 0 4px 20px rgba(79,70,229,0.45) !important;
    transition: opacity 0.2s, transform 0.15s !important;
}
.stButton > button[kind="primary"]:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}

.stDownloadButton > button {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-weight: 600 !important;
}
.stDownloadButton > button:hover {
    background: rgba(255,255,255,0.13) !important;
}

.stProgress > div > div {
    background: linear-gradient(90deg, #4f46e5, #7c3aed) !important;
    border-radius: 999px;
}

.streamlit-expanderHeader {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 10px !important;
    color: #94a3b8 !important;
}

hr { border-color: rgba(255,255,255,0.08) !important; }
img { border-radius: 12px !important; }

.stCodeBlock {
    border-radius: 10px !important;
    background: rgba(0,0,0,0.3) !important;
}

.stSpinner > div { border-top-color: #7c3aed !important; }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# =====================================================
# HERO
# =====================================================

st.markdown("""
<div class="hero">
    <h1>🎥 YouTube Summarizer</h1>
    <p>Paste any YouTube link — get a clean AI-powered summary instantly</p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# CONSTANTS
# =====================================================

CHUNK_SIZE      = 6_000
CHUNK_OVERLAP   = 100
SHORT_THRESHOLD = 20_000
MAX_FINAL_CHARS = 10_000
MAX_KW_CHARS    = 1_200

OUTPUT_LANGUAGES = [
    "English",
    "Hindi", "Telugu", "Tamil", "Kannada", "Malayalam",
    "Bengali", "Marathi", "Gujarati", "Punjabi",
    "Spanish", "French", "German", "Italian", "Portuguese",
    "Arabic", "Turkish", "Persian", "Urdu",
    "Chinese (Simplified)", "Chinese (Traditional)",
    "Japanese", "Korean",
    "Russian", "Ukrainian", "Polish",
    "Swahili", "Vietnamese", "Thai", "Nepali", "Sinhala",
    "Other (type below)"
]

SCRIPT_FONTS = {
    "devanagari": (
        "NotoSansDevanagari",
        "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/"
        "NotoSansDevanagari/NotoSansDevanagari-Regular.ttf"
    ),
    "telugu": (
        "NotoSansTelugu",
        "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/"
        "NotoSansTelugu/NotoSansTelugu-Regular.ttf"
    ),
    "tamil": (
        "NotoSansTamil",
        "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/"
        "NotoSansTamil/NotoSansTamil-Regular.ttf"
    ),
    "kannada": (
        "NotoSansKannada",
        "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/"
        "NotoSansKannada/NotoSansKannada-Regular.ttf"
    ),
    "malayalam": (
        "NotoSansMalayalam",
        "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/"
        "NotoSansMalayalam/NotoSansMalayalam-Regular.ttf"
    ),
    "bengali": (
        "NotoSansBengali",
        "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/"
        "NotoSansBengali/NotoSansBengali-Regular.ttf"
    ),
    "gujarati": (
        "NotoSansGujarati",
        "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/"
        "NotoSansGujarati/NotoSansGujarati-Regular.ttf"
    ),
    "gurmukhi": (
        "NotoSansGurmukhi",
        "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/"
        "NotoSansGurmukhi/NotoSansGurmukhi-Regular.ttf"
    ),
    "arabic": (
        "NotoSansArabic",
        "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/"
        "NotoSansArabic/NotoSansArabic-Regular.ttf"
    ),
    "cjk": (
        "NotoSansCJK",
        "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/"
        "Japanese/NotoSansCJKjp-Regular.otf"
    ),
    "korean": (
        "NotoSansKR",
        "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/"
        "Korean/NotoSansCJKkr-Regular.otf"
    ),
    "cyrillic": (
        "NotoSans",
        "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/"
        "NotoSans/NotoSans-Regular.ttf"
    ),
    "latin": ("Helvetica", None),
}


# =====================================================
# HELPERS
# =====================================================

def detect_script_family(text: str) -> str:
    counts = {k: 0 for k in SCRIPT_FONTS}
    for ch in text[:500]:
        cp = ord(ch)
        if   0x0900 <= cp <= 0x097F: counts["devanagari"] += 1
        elif 0x0C00 <= cp <= 0x0C7F: counts["telugu"]     += 1
        elif 0x0B80 <= cp <= 0x0BFF: counts["tamil"]      += 1
        elif 0x0C80 <= cp <= 0x0CFF: counts["kannada"]    += 1
        elif 0x0D00 <= cp <= 0x0D7F: counts["malayalam"]  += 1
        elif 0x0980 <= cp <= 0x09FF: counts["bengali"]    += 1
        elif 0x0A80 <= cp <= 0x0AFF: counts["gujarati"]   += 1
        elif 0x0A00 <= cp <= 0x0A7F: counts["gurmukhi"]   += 1
        elif 0x0600 <= cp <= 0x06FF: counts["arabic"]     += 1
        elif 0x4E00 <= cp <= 0x9FFF: counts["cjk"]        += 1
        elif 0xAC00 <= cp <= 0xD7AF: counts["korean"]     += 1
        elif 0x0400 <= cp <= 0x04FF: counts["cyrillic"]   += 1
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else "latin"


# ✅ FIXED — deduplicate and limit to 12
def parse_bullets(text: str) -> list:
    lines  = text.strip().split("\n")
    result = []
    seen   = set()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        line = re.sub(r'^[-•*]\s*', '', line).strip()
        if not line:
            continue
        # ✅ Deduplicate — skip if already seen
        normalised = re.sub(r'\s+', ' ', line.lower())
        if normalised in seen:
            continue
        seen.add(normalised)
        result.append(line)
        # ✅ Hard cap at 12
        if len(result) >= 12:
            break
    return result


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
# FONT DOWNLOADER
# =====================================================

@st.cache_resource
def get_font_for_script(script_family: str):
    font_name, url = SCRIPT_FONTS.get(script_family, SCRIPT_FONTS["latin"])
    if url is None:
        return font_name, None
    font_filename = url.split("/")[-1]
    if not os.path.exists(font_filename):
        r = req_lib.get(url)
        r.raise_for_status()
        with open(font_filename, "wb") as f:
            f.write(r.content)
    return font_name, font_filename


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
# FETCH TRANSCRIPT
# =====================================================

@st.cache_data(show_spinner=False)
def get_video_transcript(video_id: str):
    attempts = [
        {"videoId": video_id, "lang": "en"},
        {"videoId": video_id},
    ]
    for params in attempts:
        r = req_lib.get(
            "https://api.supadata.ai/v1/youtube/transcript",
            params=params,
            headers={"x-api-key": st.secrets["SUPADATA_API_KEY"]}
        )
        data = r.json()
        if "content" in data and data["content"]:
            plain_text = " ".join([s["text"] for s in data["content"]])
            return plain_text
    raise Exception(
        "No English transcript found. "
        "Please use a video with English captions enabled."
    )


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
# SPLIT TRANSCRIPT
# =====================================================

def split_transcript(text: str) -> list:
    if len(text) <= SHORT_THRESHOLD:
        return [text]
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_text(text)


# =====================================================
# MAP STEP
# =====================================================

def map_chunk(chunk: str, llm, idx: int, total: int) -> str:
    prompt = (
        f"Section {idx + 1} of {total}.\n"
        f"Extract the most important key points from this transcript section.\n"
        f"Output ONLY short English bullet points starting with '-'.\n"
        f"Be specific: include names, numbers, facts, and actions.\n"
        f"No intro, no outro — only bullets.\n\n"
        f"{chunk}"
    )
    for attempt in range(4):
        try:
            time.sleep(12)
            return llm.invoke([HumanMessage(content=prompt)]).content
        except Exception as e:
            err = str(e)
            if "429" in err or "413" in err or "rate_limit" in err:
                time.sleep(20 * (attempt + 1))
            else:
                raise
    raise RuntimeError(f"Section {idx + 1} failed after 4 attempts.")


# =====================================================
# REDUCE STEP
# =====================================================

# ✅ FIXED — "UP TO 12", add strict no-repeat rule
def reduce_to_summary(bullets: str, language: str, llm) -> str:
    if language.lower() != "english":
        lang_top = (
            f"YOU MUST RESPOND ONLY IN {language.upper()}. "
            f"DO NOT USE ENGLISH AT ALL. "
            f"YOUR ENTIRE RESPONSE MUST BE IN {language.upper()}.\n\n"
        )
        lang_bottom = (
            f"\n\nREMINDER: Write every bullet in {language} only. "
            f"No English."
        )
    else:
        lang_top    = ""
        lang_bottom = ""

    prompt = (
        f"{lang_top}"
        f"Below are key points extracted from a YouTube video transcript.\n"
        f"Consolidate them into UP TO 12 final bullet points "  # ✅ UP TO not EXACTLY
        f"that best summarise the entire video.\n"
        f"Rules:\n"
        f"- Remove ALL duplicates — every bullet must be unique\n"  # ✅ explicit
        f"- NEVER repeat the same point twice\n"                    # ✅ explicit
        f"- Keep only the most important and unique facts\n"
        f"- Each bullet must be one concise line\n"
        f"- Start each bullet with '-'\n"
        f"- If you run out of unique points, stop — do NOT repeat\n"  # ✅ key fix
        f"- No intro sentence, no outro sentence\n"
        f"- Only output the bullet points"
        f"{lang_bottom}\n\n"
        f"Key points:\n{bullets}"
    )

    for attempt in range(4):
        try:
            time.sleep(12)
            return llm.invoke([HumanMessage(content=prompt)]).content
        except Exception as e:
            err = str(e)
            if "429" in err or "413" in err or "rate_limit" in err:
                time.sleep(20 * (attempt + 1))
            else:
                raise
    raise RuntimeError("Failed to generate final summary.")


# =====================================================
# EXTRACT KEYWORDS
# =====================================================

def extract_keywords(text: str, llm, language: str) -> list:
    time.sleep(12)
    if language.lower() != "english":
        lang_note = (
            f"IMPORTANT: Return the keywords translated and written "
            f"in {language} only. Do not use English words."
        )
    else:
        lang_note = ""

    prompt = (
        f"Extract exactly 8 important keywords or key phrases from this content.\n"
        f"Return ONLY a comma-separated list, nothing else.\n"
        f"{lang_note}\n\n"
        f"{text[:MAX_KW_CHARS]}"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    keywords = [k.strip() for k in response.content.split(",")]
    return [k for k in keywords if k][:8]


# =====================================================
# GENERATE PDF
# =====================================================

def create_pdf(video_id: str, keywords: list,
               summary_bullets: list, language: str) -> bytes:

    all_text     = " ".join(summary_bullets)
    script       = detect_script_family(all_text)
    fname, fpath = get_font_for_script(script)

    pdf = FPDF()
    pdf.add_page()
    if fpath:
        pdf.add_font(fname, style="", fname=fpath)

    def sf(style_flag="", size=10):
        if fpath:
            pdf.set_font(fname, size=size)
        else:
            pdf.set_font("Helvetica", style=style_flag, size=size)

    # Title
    sf("B", 18)
    pdf.cell(0, 12, "YouTube Video Summary", ln=True, align="C")
    pdf.ln(2)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)

    # Metadata
    sf("", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 5, f"Source: https://youtube.com/watch?v={video_id}", ln=True)
    pdf.cell(0, 5, f"Language: {language}", ln=True)
    pdf.ln(5)

    # Keywords
    if keywords:
        sf("B", 12)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 8, "Keywords:", ln=True)
        sf("", 10)
        pdf.set_text_color(60, 60, 60)
        pdf.multi_cell(0, 6, ", ".join(keywords))
        pdf.ln(4)

    # Summary bullets
    if summary_bullets:
        sf("B", 12)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 8, "Video Summary:", ln=True)
        pdf.ln(2)
        sf("", 10)
        pdf.set_text_color(30, 30, 30)
        for bullet in summary_bullets:
            clean = re.sub(r'\*\*(.+?)\*\*', r'\1', bullet)
            clean = re.sub(r'\*(.+?)\*',     r'\1', clean)
            pdf.multi_cell(0, 7, f"- {clean}")
            pdf.ln(1)

    return bytes(pdf.output())


# =====================================================
# LANGUAGE SELECTOR
# =====================================================

_, col_lang, _ = st.columns([1, 2, 1])
with col_lang:
    selected_lang = st.selectbox(
        "🌐 Output language",
        OUTPUT_LANGUAGES,
        help="Transcript is always fetched in English then translated."
    )

if selected_lang == "Other (type below)":
    _, col_other, _ = st.columns([1, 2, 1])
    with col_other:
        language = st.text_input(
            "Type your language",
            placeholder="e.g. Swahili, Vietnamese, Thai …"
        ).strip()
    if not language:
        st.info("Please type a language name above to continue.")
        st.stop()
else:
    language = selected_lang

if language.lower() != "english":
    st.markdown(
        f'<div class="info-banner">ℹ️ Transcript is fetched in <b>English</b> '
        f'and the summary will be translated to <b>{language}</b>.</div>',
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)


# =====================================================
# URL INPUT
# =====================================================

youtube_url = st.text_input(
    "🔗 YouTube Video URL",
    placeholder="https://www.youtube.com/watch?v=...",
)

if youtube_url.strip():
    preview_id = extract_video_id(youtube_url.strip())
    if preview_id:
        col_img, col_meta = st.columns([1, 2])
        with col_img:
            st.image(
                f"https://img.youtube.com/vi/{preview_id}/mqdefault.jpg",
                use_container_width=True
            )
        with col_meta:
            st.markdown(f"""
            <div class="video-meta">
                <div style="color:#34d399;font-weight:700;
                            font-size:0.9rem;margin-bottom:0.5rem;">
                    ✅ Valid YouTube URL detected
                </div>
                <div style="margin-bottom:0.4rem;">
                    Video ID:&nbsp;
                    <span style="background:rgba(255,255,255,0.08);
                            color:#e2e8f0;padding:2px 8px;
                            border-radius:4px;font-size:0.83rem;
                            font-family:monospace;">
                        {preview_id}
                    </span>
                </div>
                <div>
                    Output language:&nbsp;
                    <b style="color:#a78bfa">{language}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.error("⚠️ Could not recognise this YouTube URL.")

st.markdown("<br>", unsafe_allow_html=True)


# =====================================================
# GENERATE BUTTON
# =====================================================

generate = st.button(
    "🚀 Generate Summary",
    type="primary",
    use_container_width=True
)


# =====================================================
# BUTTON ACTION
# =====================================================

if generate:
    if not youtube_url.strip():
        st.warning("Please enter a YouTube URL.")
        st.stop()

    try:
        video_id = extract_video_id(youtube_url)
        if not video_id:
            st.error("Invalid YouTube URL — please check and try again.")
            st.stop()

        # Step 1: Fetch transcript
        with st.spinner("🔍 Fetching transcript…"):
            raw_transcript = get_video_transcript(video_id)

        # Step 2: Compress + trim
        transcript = compress_transcript(raw_transcript)
        transcript = transcript[:40_000]

        # Step 3: Split into chunks
        chunks   = split_transcript(transcript)
        llm      = load_llm()
        progress = st.progress(0, text="⚙️ Starting…")

        # ── SHORT VIDEO — single direct call ──────────────────────────
        if len(chunks) == 1:
            progress.progress(30, text="⚙️ Summarising…")
            raw_summary = reduce_to_summary(
                chunks[0][:MAX_FINAL_CHARS], language, llm
            )

        # ── LONG VIDEO — map → reduce ─────────────────────────────────
        else:
            map_results = [""] * len(chunks)
            completed   = 0

            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = {
                    executor.submit(
                        map_chunk, chunk, llm, i, len(chunks)
                    ): i
                    for i, chunk in enumerate(chunks)
                }
                for future in as_completed(futures):
                    i = futures[future]
                    map_results[i] = future.result()
                    completed += 1
                    pct = int((completed / len(chunks)) * 60)
                    progress.progress(
                        pct,
                        text=f"⚙️ Processing section {completed}/{len(chunks)}…"
                    )

            progress.progress(68, text="⚙️ Consolidating summary…")
            all_bullets = "\n".join(map_results)
            raw_summary = reduce_to_summary(
                all_bullets[:MAX_FINAL_CHARS], language, llm
            )

        # Step 4: Keywords
        progress.progress(85, text="🔑 Extracting keywords…")
        keywords = extract_keywords(raw_summary, llm, language)

        progress.progress(100, text="✅ Done!")
        time.sleep(0.3)
        progress.empty()

        # Parse bullets
        summary_bullets = parse_bullets(raw_summary)

        # ── KEYWORDS ──────────────────────────────────────────────────
        kw_html = " ".join([
            f'<span class="kw-pill">{k}</span>' for k in keywords
        ])
        st.markdown(f"""
        <div class="glass-card">
            <div class="section-heading">🔑 Keywords</div>
            {kw_html}
        </div>
        """, unsafe_allow_html=True)

        # ── SUMMARY ───────────────────────────────────────────────────
        st.markdown("""
        <div class="glass-card">
            <div class="section-heading">📌 Video Summary</div>
        </div>
        """, unsafe_allow_html=True)

        for bullet in summary_bullets:
            st.markdown(f"- {bullet}")

        # ── DOWNLOADS ─────────────────────────────────────────────────
        txt_out  = "VIDEO SUMMARY\n" + "=" * 40 + "\n\n"
        txt_out += "Keywords: " + ", ".join(keywords) + "\n\n"
        txt_out += "SUMMARY:\n\n"
        for b in summary_bullets:
            txt_out += f"- {b}\n"

        st.markdown(
            '<p style="color:#e2e8f0;font-weight:700;'
            'font-size:1rem;margin-top:1.2rem;">⬇️ Export</p>',
            unsafe_allow_html=True
        )
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                "📋 Download as Text", txt_out,
                "summary.txt", "text/plain",
                use_container_width=True
            )
        with col_dl2:
            pdf_bytes = create_pdf(
                video_id, keywords, summary_bullets, language
            )
            st.download_button(
                "📄 Download as PDF", pdf_bytes,
                "summary.pdf", "application/pdf",
                use_container_width=True
            )
        with st.expander("📋 View / copy raw text"):
            st.code(txt_out, language=None)

    except Exception as e:
        st.markdown(f"""
        <div style="background:rgba(239,68,68,0.12);
                    border:1px solid rgba(239,68,68,0.35);
                    border-radius:12px;padding:1rem;color:#fca5a5;
                    margin-top:1rem;">
            ❌ <b>Error:</b> {e}
        </div>
        """, unsafe_allow_html=True)