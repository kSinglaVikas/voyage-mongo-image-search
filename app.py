"""Streamlit app: search images by text or by a query image using MongoDB Atlas Vector Search."""
from pathlib import Path

import streamlit as st
import voyageai
from PIL import Image
from pymongo import MongoClient

import config

DATA_DIR = Path(__file__).parent / "data"


@st.cache_resource
def get_clients():
    vo = voyageai.Client(api_key=config.VOYAGE_API_KEY)
    coll = MongoClient(config.MONGODB_URI)[config.MONGODB_DB][config.MONGODB_COLLECTION]
    return vo, coll


@st.cache_data
def get_labels() -> list[str]:
    _, coll = get_clients()
    return sorted(coll.distinct("label"))


def embed_query(vo, query) -> list[float]:
    result = vo.multimodal_embed(
        inputs=[[query]],
        model=config.VOYAGE_MODEL,
        input_type="query",
    )
    return result.embeddings[0]


def vector_search(coll, vector: list[float], k: int, label: str | None):
    stage = {
        "index": config.VECTOR_INDEX_NAME,
        "path": "embedding",
        "queryVector": vector,
        "numCandidates": k * 20,
        "limit": k,
    }
    if label:
        stage["filter"] = {"label": label}
    pipeline = [
        {"$vectorSearch": stage},
        {"$project": {"_id": 0, "path": 1, "label": 1, "score": {"$meta": "vectorSearchScore"}}},
    ]
    return list(coll.aggregate(pipeline))


def main() -> None:
    st.set_page_config(page_title="Image Search", layout="wide")
    st.title("Image Vector Search")
    vo, coll = get_clients()

    with st.sidebar:
        k = st.slider("Results", 1, 30, 9)
        label = st.selectbox("Filter by label", ["(all)", *get_labels()])
        label = None if label == "(all)" else label

    tab_text, tab_image = st.tabs(["Text query", "Image query"])
    query = None
    with tab_text:
        text = st.text_input("Describe the image", placeholder="ancient ruins on a mountain")
        if st.button("Search", key="text_btn") and text.strip():
            query = text.strip()
    with tab_image:
        upload = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
        if upload:
            img = Image.open(upload).convert("RGB")
            img.thumbnail((1024, 1024))
            st.image(img, width=250)
            if st.button("Search", key="image_btn"):
                query = img

    if query is None:
        return

    with st.spinner("Searching..."):
        results = vector_search(coll, embed_query(vo, query), k, label)

    if not results:
        st.info("No results.")
        return
    cols = st.columns(3)
    for i, r in enumerate(results):
        with cols[i % 3]:
            st.image(str(DATA_DIR / r["path"]), width="stretch")
            st.caption(f"{r['label']} — score {r['score']:.3f}")


if __name__ == "__main__":
    main()
