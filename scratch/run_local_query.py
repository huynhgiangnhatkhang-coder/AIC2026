import sys
import yaml
import torch
import warnings

warnings.filterwarnings("ignore")

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

from src.retrieval.milvus_retriever import MilvusRetriever
from src.query.florence_kis import FlorenceKISSearcher

milvus_retriever = MilvusRetriever(
    db_path=cfg["index"]["milvus_db_path"],
    collection_name=cfg["index"]["milvus_collection"],
    model_name=cfg["clip"]["model_name"],
    device=cfg["clip"]["device"]
)

searcher = FlorenceKISSearcher(
    vector_retriever=milvus_retriever,
    collection_name=cfg["index"]["milvus_collection"],
    keyframes_dir=str(cfg["data"].get("keyframes_root", "DATASET")),
    ocr_db_path="ocr_database.json",
    max_answers=25,
    batch_size=32
)

query = "vạch đích của cuộc đua xe đạp, 1 tay đua áo vàng quần đen, 1 tay đua áo xanh dương quần đen và 1 tay đua áo xanh dương quần đỏ"
print("RUNNING QUERY:", query)
results = searcher.search(query, search_mode="hybrid", top_k=25)

print("\n--- TOP 25 RESULTS ---")
for i, res in enumerate(results):
    print(f"{i+1}. {res['video_id']} - Frame {res['frame_id']} - Score: {res.get('score', 0):.4f}")
