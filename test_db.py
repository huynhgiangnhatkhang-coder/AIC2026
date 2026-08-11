from pymilvus import MilvusClient
client = MilvusClient("aic_kis_database_siglip.db")
client.load_collection("kis_keyframes_siglip")
res = client.query("kis_keyframes_siglip", filter="video_id == 'L22_V002.mp4' and frame_id == 209", output_fields=["video_id", "frame_id"])
print("Query result:", res)
