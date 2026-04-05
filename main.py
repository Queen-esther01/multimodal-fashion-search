import os
import ssl
import time
import urllib.request
from io import BytesIO

import certifi
import numpy as np
import chromadb
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader

load_dotenv()

db = "qdrant"  # "qdrant" or "chroma"

st.set_page_config(
    page_title="Multi-Modal Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, p, h1, h2, h3, h4, h5, h6, span, div, button, input, textarea, label, a {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background: #0A0D12;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    div[data-testid="stSidebarContent"] {
        background: #0E1117;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Hero header */
    .hero {
        text-align: center;
        padding: 2.5rem 0 1.5rem;
    }
    .hero h1 {
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6C63FF 0%, #B794F6 50%, #6C63FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }
    .hero p {
        color: #8B949E;
        font-size: 1.05rem;
        font-weight: 300;
        letter-spacing: 0.2px;
    }

    /* Search mode toggle */
    .search-toggle {
        display: flex;
        justify-content: center;
        gap: 0;
        margin: 1.5rem auto 2rem;
        background: #161B22;
        border-radius: 12px;
        padding: 4px;
        width: fit-content;
        border: 1px solid #21262D;
    }
    .toggle-btn {
        padding: 10px 28px;
        border-radius: 9px;
        font-size: 0.88rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s ease;
        border: none;
        color: #8B949E;
        background: transparent;
        letter-spacing: 0.3px;
    }
    .toggle-btn.active {
        background: #6C63FF;
        color: #FFFFFF;
        box-shadow: 0 2px 12px rgba(108, 99, 255, 0.3);
    }

    /* Input area */
    .stTextInput > div > div > input {
        background: #161B22 !important;
        border: 1px solid #21262D !important;
        border-radius: 10px !important;
        color: #E6EDF3 !important;
        padding: 14px 18px !important;
        font-size: 0.95rem !important;
        transition: border-color 0.2s ease !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #6C63FF !important;
        box-shadow: 0 0 0 3px rgba(108, 99, 255, 0.15) !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: #484F58 !important;
    }
    .stTextInput label {
        display: none !important;
    }


    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6C63FF, #8B5CF6) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 8px 20px !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        letter-spacing: 0.3px !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 14px rgba(108, 99, 255, 0.25) !important;
        min-height: 0 !important;
        height: auto !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(108, 99, 255, 0.35) !important;
    }
    .stButton > button:active {
        transform: translateY(0px) !important;
    }

    /* Result card */
    .result-card {
        background: #161B22;
        border: 1px solid #21262D;
        border-radius: 14px;
        overflow: hidden;
        transition: all 0.25s ease;
        height: 100%;
    }
    .result-card:hover {
        border-color: #6C63FF;
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(108, 99, 255, 0.12);
    }
    .result-card img {
        width: 100%;
        height: 220px;
        object-fit: cover;
        display: block;
    }
    .card-body {
        padding: 16px;
    }
    .card-title {
        font-size: 0.92rem;
        font-weight: 600;
        color: #E6EDF3;
        margin-bottom: 10px;
        line-height: 1.35;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    .card-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-bottom: 12px;
    }
    .tag {
        background: #21262D;
        color: #8B949E;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 0.3px;
        text-transform: uppercase;
    }
    .tag.color-tag {
        border-left: 3px solid #6C63FF;
    }
    .score-bar-container {
        margin-top: 10px;
    }
    .score-label {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 5px;
    }
    .score-text {
        font-size: 0.72rem;
        color: #484F58;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 500;
    }
    .score-value {
        font-size: 0.82rem;
        font-weight: 700;
        color: #6C63FF;
    }
    .score-bar {
        height: 4px;
        background: #21262D;
        border-radius: 2px;
        overflow: hidden;
    }
    .score-fill {
        height: 100%;
        border-radius: 2px;
        background: linear-gradient(90deg, #6C63FF, #B794F6);
        transition: width 0.5s ease;
    }

    /* Uploaded image preview */
    .preview-container {
        background: #161B22;
        border: 1px solid #21262D;
        border-radius: 14px;
        padding: 12px;
        margin-bottom: 1.5rem;
    }
    .preview-container img {
        border-radius: 10px;
        width: 100%;
        max-height: 300px;
        object-fit: contain;
    }
    .preview-label {
        text-align: center;
        color: #484F58;
        font-size: 0.78rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 10px;
    }

    /* Section header */
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 2rem 0 1.2rem;
        padding-bottom: 12px;
        border-bottom: 1px solid #21262D;
    }
    .section-header h2 {
        font-size: 1.2rem;
        font-weight: 600;
        color: #E6EDF3;
        margin: 0;
    }
    .result-count {
        background: #6C63FF;
        color: white;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 20px;
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: #6C63FF !important;
    }
    .stSpinner > div > div {
        color: #8B949E !important;
    }

    /* Radio buttons → hidden, we use custom toggle */
    div[data-testid="stRadio"] {
        display: none;
    }

    /* Divider */
    hr {
        border-color: #21262D !important;
    }

    /* Tabs override (used for search mode) */
    .stTabs [data-baseweb="tab-list"] {
        background: #161B22;
        border-radius: 10px;
        padding: 3px;
        gap: 0;
        border: 1px solid #21262D;
        justify-content: center;
        width: fit-content;
        margin: 0 auto;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 7px;
        color: #8B949E;
        font-weight: 500;
        padding: 6px 18px;
        font-size: 0.8rem;
        background: transparent;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background: #6C63FF !important;
        color: #FFFFFF !important;
        box-shadow: 0 2px 12px rgba(108, 99, 255, 0.3);
    }
    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1.5rem;
    }

    /* Column gap fix */
    div[data-testid="stHorizontalBlock"] {
        gap: 1rem;
    }

    /* Image in columns */
    div[data-testid="stImage"] {
        border-radius: 10px;
        overflow: hidden;
    }

    /* Hide default streamlit elements */
    #MainMenu, footer, .stDeployButton {
        display: none !important;
    }

    /* Footer */
    .app-footer {
        text-align: center;
        padding: 3rem 0 1.5rem;
        border-top: 1px solid #21262D;
        margin-top: 3rem;
    }
    .app-footer p {
        color: #484F58;
        font-size: 0.78rem;
        margin: 0;
    }
    .app-footer a {
        color: #6C63FF;
        text-decoration: none;
        font-weight: 500;
    }
    .app-footer a:hover {
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)


IMAGE_DIR = "images"
R2_URL = (os.getenv("R2_URL") or "").rstrip("/")
TEXT_INPUT_KEY = "text_search"
TEXT_RESULTS_KEY = "_active_text_query"
SUGGESTION_PILLS_KEY = "sug_pills"
LAST_SUGGESTION_KEY = "_last_suggestion"
IMAGE_TAB_SWITCH_NONCE_KEY = "_image_tab_switch_nonce"


def _apparels_filename(record_id: str, uri: str | None) -> str:
    if uri:
        base = os.path.basename(str(uri).replace("\\", "/").rstrip("/"))
        if base and "." in base:
            return base
    rid = str(record_id)
    if "." in rid:
        return rid
    return f"{rid}.jpg"


def _resolve_qdrant_image_ref(point_id) -> str:
    filename = _apparels_filename(str(point_id), None)
    if R2_URL:
        return f"{R2_URL}/apparels/{filename}"
    return f"{IMAGE_DIR}/{filename}"


def _resolve_chroma_image_ref(record_id: str, uri: str) -> str:
    if R2_URL:
        filename = _apparels_filename(record_id, uri)
        return f"{R2_URL}/apparels/{filename}"
    return uri


def _image_ref_available(ref: str) -> bool:
    if not ref:
        return False
    if ref.startswith(("http://", "https://")):
        return True
    return os.path.exists(ref)


def _load_pil_rgb(ref: str) -> Image.Image:
    if ref.startswith(("http://", "https://")):
        request = urllib.request.Request(
            ref,
            headers={"User-Agent": "streamlit-image-search/1.0"},
        )
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(request, context=ssl_context, timeout=20) as resp:
            return Image.open(BytesIO(resp.read())).convert("RGB")
    return Image.open(ref).convert("RGB")


# ── Chroma ──

@st.cache_resource
def get_chroma_collection():
    client = chromadb.CloudClient(
        api_key=os.getenv("CHROMA_API_KEY"),
        tenant=os.getenv("CHROMA_TENANT"),
        database=os.getenv("CHROMA_DATABASE"),
    )
    return client.get_or_create_collection(
        name=os.getenv("COLLECTION_NAME"),
        embedding_function=OpenCLIPEmbeddingFunction(),
        data_loader=ImageLoader(),
    )


def _normalize_chroma(results: dict) -> list[dict]:
    if not results or not results["ids"] or not results["ids"][0]:
        return []
    return [
        {
            "id": results["ids"][0][i],
            "metadata": results["metadatas"][0][i],
            "score": max(0, (1 - results["distances"][0][i]) * 100),
            "img_path": _resolve_chroma_image_ref(
                results["ids"][0][i],
                results["uris"][0][i],
            ),
        }
        for i in range(len(results["ids"][0]))
    ]


def chroma_search_by_text(query_text: str) -> list[dict]:
    collection = get_chroma_collection()
    return _normalize_chroma(collection.query(
        query_texts=[query_text],
        n_results=5,
        include=["uris", "distances", "metadatas"],
    ))


def chroma_search_by_image(image_array: np.ndarray) -> list[dict]:
    collection = get_chroma_collection()
    return _normalize_chroma(collection.query(
        query_images=[image_array],
        n_results=5,
        include=["uris", "distances", "metadatas"],
    ))


# ── Qdrant ──

@st.cache_resource
def get_qdrant_client():
    return QdrantClient(
        url=os.getenv("QDRANT_ENDPOINT"),
        port=6333,
        api_key=os.getenv("QDRANT_API_KEY"),
    )


@st.cache_resource
def get_model():
    return SentenceTransformer("clip-ViT-B-32")


def _normalize_qdrant(results) -> list[dict]:
    if not results or not results.points:
        return []
    return [
        {
            "id": point.id,
            "metadata": point.payload,
            "score": max(0, point.score * 100),
            "img_path": _resolve_qdrant_image_ref(point.id),
        }
        for point in results.points
    ]


def qdrant_search_by_text(query_text: str) -> list[dict]:
    model = get_model()
    client = get_qdrant_client()
    embedding = model.encode(query_text)
    return _normalize_qdrant(client.query_points(
        collection_name=os.getenv("COLLECTION_NAME"),
        query=embedding.tolist(),
        limit=5,
    ))


def qdrant_search_by_image(image_array: np.ndarray) -> list[dict]:
    model = get_model()
    client = get_qdrant_client()
    embedding = model.encode(Image.fromarray(image_array))
    return _normalize_qdrant(client.query_points(
        collection_name=os.getenv("COLLECTION_NAME"),
        query=embedding.tolist(),
        limit=5,
    ))


# ── Dispatch ──

def search_by_text(query_text: str) -> list[dict]:
    if db == "chroma":
        return chroma_search_by_text(query_text)
    return qdrant_search_by_text(query_text)


def search_by_image(image_array: np.ndarray) -> list[dict]:
    if db == "chroma":
        return chroma_search_by_image(image_array)
    return qdrant_search_by_image(image_array)


def _set_active_text_query(query: str, *, clear_suggestion: bool) -> None:
    normalized_query = query.strip()
    if not normalized_query:
        st.session_state.pop(TEXT_RESULTS_KEY, None)
        if clear_suggestion:
            st.session_state[SUGGESTION_PILLS_KEY] = None
            st.session_state[LAST_SUGGESTION_KEY] = None
        return

    st.session_state[TEXT_RESULTS_KEY] = normalized_query
    st.session_state[TEXT_INPUT_KEY] = normalized_query

    if clear_suggestion:
        st.session_state[SUGGESTION_PILLS_KEY] = None
        st.session_state[LAST_SUGGESTION_KEY] = None


def submit_text_search() -> None:
    _set_active_text_query(
        st.session_state.get(TEXT_INPUT_KEY, ""),
        clear_suggestion=True,
    )


def submit_suggestion_search() -> None:
    selected_suggestion = st.session_state.get(SUGGESTION_PILLS_KEY)

    if not selected_suggestion:
        st.session_state[LAST_SUGGESTION_KEY] = None
        return

    if selected_suggestion == st.session_state.get(LAST_SUGGESTION_KEY):
        return

    st.session_state[LAST_SUGGESTION_KEY] = selected_suggestion
    _set_active_text_query(selected_suggestion, clear_suggestion=False)


def render_result_card(metadata: dict, score: float) -> str:
    similarity = score

    name = metadata.get("name", "Unknown Item")
    article = metadata.get("article_type", "")
    color = metadata.get("base_color", "")
    season = metadata.get("season", "")
    usage = metadata.get("usage", "")
    sub_cat = metadata.get("sub_category", "")

    tags_html = ""
    if article:
        tags_html += f'<span class="tag">{article}</span>'
    if color:
        tags_html += f'<span class="tag color-tag">{color}</span>'
    if season:
        tags_html += f'<span class="tag">{season}</span>'
    if usage:
        tags_html += f'<span class="tag">{usage}</span>'

    return f"""
    <div class="result-card">
        <div class="card-body">
            <div class="card-title">{name}</div>
            <div class="card-tags">{tags_html}</div>
            <div class="score-bar-container">
                <div class="score-label">
                    <span class="score-text">Match</span>
                    <span class="score-value">{similarity:.1f}%</span>
                </div>
                <div class="score-bar">
                    <div class="score-fill" style="width: {similarity}%"></div>
                </div>
            </div>
        </div>
    </div>
    """


# ── Header ──
st.markdown("""
<div class="hero">
    <h1>Multi-Modal Search</h1>
    <p>Find fashion items by image or text</p>
</div>
""", unsafe_allow_html=True)

# ── Search Tabs ──
tab_text, tab_image = st.tabs(["Text Search", "Image Search"])

with tab_text:
    col_input, col_btn = st.columns([6, 1], vertical_alignment="bottom")
    with col_input:
        st.text_input(
            "Search",
            placeholder="Describe what you're looking for…  e.g. black summer dress",
            key=TEXT_INPUT_KEY,
        )
    with col_btn:
        st.button(
            "Search",
            key="text_btn",
            use_container_width=True,
            on_click=submit_text_search,
        )

    suggestions = [
        "A black elegant dress",
        "Casual summer outfit",
        "Blue denim jeans",
        "Red formal shirt",
    ]
    st.pills(
        "Suggestions",
        suggestions,
        key=SUGGESTION_PILLS_KEY,
        on_change=submit_suggestion_search,
    )

    search_term = st.session_state.get(TEXT_RESULTS_KEY, "").strip()

    if search_term:
        with st.spinner("Searching the collection…"):
            items = search_by_text(search_term)

        if items:
            st.markdown(f"""
            <div class="section-header">
                <h2>Results</h2>
                <span class="result-count">Top {len(items)} results</span>
            </div>
            """, unsafe_allow_html=True)

            cols_per_row = 3
            for row_start in range(0, len(items), cols_per_row):
                cols = st.columns(cols_per_row)
                for i, col in enumerate(cols):
                    idx = row_start + i
                    if idx >= len(items):
                        break
                    with col:
                        item = items[idx]
                        if _image_ref_available(item["img_path"]):
                            st.image(item["img_path"], use_container_width=True)
                        st.markdown(
                            render_result_card(item["metadata"], item["score"]),
                            unsafe_allow_html=True,
                        )
                        if _image_ref_available(item["img_path"]):
                            if st.button("Find similar", key=f"sim_{idx}", use_container_width=True):
                                st.session_state["_result_img_path"] = item["img_path"]
                                st.session_state["_switch_to_image_tab"] = True
                                st.session_state[IMAGE_TAB_SWITCH_NONCE_KEY] = time.time_ns()
                                st.rerun()
        else:
            st.markdown(
                '<p style="text-align:center; color:#484F58; padding:3rem 0;">No results found. Try a different query.</p>',
                unsafe_allow_html=True,
            )

with tab_image:
    result_img_path = st.session_state.get("_result_img_path")

    uploaded_file = st.file_uploader(
        "Drop an image here or click to upload",
        type=["jpg", "jpeg", "png", "webp"],
        key="image_upload",
    )

    if uploaded_file is not None:
        st.session_state.pop("_result_img_path", None)
        result_img_path = None
        pil_image = Image.open(uploaded_file)
        image_rgb = np.array(pil_image.convert("RGB"))
    elif result_img_path and _image_ref_available(result_img_path):
        pil_image = _load_pil_rgb(result_img_path)
        image_rgb = np.array(pil_image)
    else:
        pil_image = None
        image_rgb = None

    if image_rgb is not None:

        with st.spinner("Analyzing image and searching…"):
            items = search_by_image(image_rgb)

        if items:
            left_col, right_col = st.columns([1, 3], gap="large")

            with left_col:
                img_label = "Selected Image" if result_img_path else "Uploaded Image"
                st.markdown(f"""
                <div class="section-header" style="margin-top:0.8rem;">
                    <h2>{img_label}</h2>
                </div>
                """, unsafe_allow_html=True)
                st.image(pil_image, use_container_width=True)

            with right_col:
                st.markdown(f"""
                <div class="section-header" style="margin-top:0.8rem;">
                    <h2>Results</h2>
                    <span class="result-count">Top {len(items)} results</span>
                </div>
                """, unsafe_allow_html=True)

                cols_per_row = 3
                for row_start in range(0, len(items), cols_per_row):
                    result_cols = st.columns(cols_per_row)
                    for i, col in enumerate(result_cols):
                        idx = row_start + i
                        if idx >= len(items):
                            break
                        with col:
                            item = items[idx]
                            if _image_ref_available(item["img_path"]):
                                st.image(item["img_path"], use_container_width=True)
                            st.markdown(
                                render_result_card(item["metadata"], item["score"]),
                                unsafe_allow_html=True,
                            )
        else:
            st.markdown(
                '<p style="text-align:center; color:#484F58; padding:3rem 0;">No results found. Try a different image.</p>',
                unsafe_allow_html=True,
            )

# Switch to Image Search after tabs exist in the DOM (JS at top ran before st.tabs, so it never worked reliably)
if st.session_state.pop("_switch_to_image_tab", False):
    switch_nonce = st.session_state.get(IMAGE_TAB_SWITCH_NONCE_KEY, time.time_ns())
    components.html(
        """
    <script>
    (function () {
        const switchNonce = %d;
        window.__lastImageTabSwitchNonce = switchNonce;

        function clickImageTab() {
            const doc = window.parent.document;
            const tabs = doc.querySelectorAll('[data-testid="stTabs"] button[role="tab"]');
            if (tabs.length > 1) {
                tabs[1].click();
                return true;
            }
            const legacy = doc.querySelectorAll('button[data-baseweb="tab"]');
            if (legacy.length > 1) {
                legacy[1].click();
                return true;
            }
            return false;
        }

        function trySwitch(attempt = 0) {
            if (clickImageTab() || attempt >= 10) {
                return;
            }
            window.setTimeout(() => trySwitch(attempt + 1), 100);
        }

        window.requestAnimationFrame(() => trySwitch());
    })();
    </script>
        """ % switch_nonce,
        height=0,
    )

# ── Footer ──
st.markdown("""
<div class="app-footer">
    <p>Dataset: <a href="https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset" target="_blank">Fashion Product Images Dataset</a> on Kaggle</p>
</div>
""", unsafe_allow_html=True)
