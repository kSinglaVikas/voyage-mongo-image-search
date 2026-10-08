import os

from dotenv import load_dotenv

load_dotenv()

VOYAGE_API_KEY = os.environ["VOYAGE_API_KEY"]
MONGODB_URI = os.environ["MONGODB_URI"]
MONGODB_DB = os.getenv("MONGODB_DB", "image_search")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "images")
VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "vector_index")
VOYAGE_MODEL = os.getenv("VOYAGE_MODEL", "voyage-multimodal-3")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
