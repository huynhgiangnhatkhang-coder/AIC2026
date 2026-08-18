from pymilvus import MilvusClient
client = MilvusClient("./aic_kis_database_siglip.db")
res = client.query(collection_name="kis_keyframes_siglip", filter="frame_id >= 0", limit=1, output_fields=["video_id", "embedding"])
print("Success" if res and "embedding" in res[0] else "Failed")
print("Dim:", len(res[0]["embedding"]) if res else 0)
