import torch
import numpy as np
from pymilvus import MilvusClient
from src.query.translate import analyze_query_offline_mt
from transformers import AutoProcessor, AutoModel

client = MilvusClient("aic_kis_database_siglip.db")
client.load_collection("kis_keyframes_siglip")
query = "bản tin về tai nạn giao thông tại dak lak"
parsed = analyze_query_offline_mt(query)

processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
model = AutoModel.from_pretrained("google/siglip-base-patch16-224").eval().cuda()

text_inputs = processor(text=[parsed["clip_query"]], padding="max_length", return_tensors="pt").to("cuda")
with torch.no_grad():
    text_features = model.get_text_features(**text_inputs)
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    text_vector = text_features[0].cpu().tolist()

res = client.query("kis_keyframes_siglip", filter="video_id == 'L22_V002.mp4' and frame_id == 209", output_fields=["embedding"])
if not res:
    print("Not found in DB")
else:
    img_vector = res[0]["embedding"]
    score = np.dot(text_vector, img_vector)
    print(f"Dot product (SigLIP score) between query and L22_V002/209 is: {score}")

