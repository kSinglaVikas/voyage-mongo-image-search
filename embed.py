"""Generate Voyage AI multimodal embeddings for images in ./data and store them in MongoDB."""
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import voyageai
from PIL import Image
from pymongo import MongoClient, UpdateOne
from pymongo.errors import OperationFailure
from pymongo.operations import SearchIndexModel
from tqdm import tqdm

import config

DATA_DIR = Path(__file__).parent / "data"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "16"))
WORKERS = int(os.getenv("EMBED_WORKERS", "8"))
MAX_SIDE = 1024


def load_image(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    return img


def ensure_vector_index(coll) -> None:
    if any(ix["name"] == config.VECTOR_INDEX_NAME for ix in coll.list_search_indexes()):
        return
    coll.create_search_index(
        SearchIndexModel(
            name=config.VECTOR_INDEX_NAME,
            type="vectorSearch",
            definition={
                "fields": [
                    {
                        "type": "vector",
                        "path": "embedding",
                        "numDimensions": config.EMBEDDING_DIM,
                        "similarity": "cosine",
                    },
                    {"type": "filter", "path": "label"},
                ]
            },
        )
    )
    print(f"Creating vector index '{config.VECTOR_INDEX_NAME}'...")
    while True:
        ix = next(iter(coll.list_search_indexes(config.VECTOR_INDEX_NAME)), None)
        if ix and ix.get("queryable"):
            break
        time.sleep(5)
    print("Vector index is ready.")


def process_batch(vo, coll, paths: list[Path]) -> int:
    batch, images = [], []
    for p in paths:
        try:
            images.append([load_image(p)])
            batch.append(p)
        except OSError as e:
            print(f"Skipping {p}: {e}")
    if not batch:
        return 0
    result = vo.multimodal_embed(
        inputs=images,
        model=config.VOYAGE_MODEL,
        input_type="document",
    )
    coll.bulk_write([
        UpdateOne(
            {"path": str(p.relative_to(DATA_DIR))},
            {"$set": {"label": p.parent.name, "embedding": emb}},
            upsert=True,
        )
        for p, emb in zip(batch, result.embeddings)
    ])
    return len(batch)


def main() -> None:
    # Retries with backoff absorb rate-limit errors from concurrent requests.
    vo = voyageai.Client(api_key=config.VOYAGE_API_KEY, max_retries=5)
    client = MongoClient(config.MONGODB_URI)
    db = client[config.MONGODB_DB]
    if config.MONGODB_COLLECTION not in db.list_collection_names():
        db.create_collection(config.MONGODB_COLLECTION)
    coll = db[config.MONGODB_COLLECTION]

    paths = sorted(p for p in DATA_DIR.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    done = set(coll.distinct("path"))
    paths = [p for p in paths if str(p.relative_to(DATA_DIR)) not in done]
    print(f"{len(paths)} images to embed ({len(done)} already stored).")

    batches = [paths[i : i + BATCH_SIZE] for i in range(0, len(paths), BATCH_SIZE)]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool, tqdm(total=len(paths), unit="img") as bar:
        futures = {pool.submit(process_batch, vo, coll, b): b for b in batches}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                print(f"Batch starting at {futures[fut][0]} failed: {e}")
            bar.update(len(futures[fut]))

    try:
        ensure_vector_index(coll)
    except OperationFailure as e:
        print(f"Could not create vector index automatically: {e}")


if __name__ == "__main__":
    main()
