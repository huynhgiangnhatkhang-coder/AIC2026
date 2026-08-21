import requests
import json

url = "http://localhost:8000/search/kis"
payload = {
    "query": "vạch đích của cuộc đua xe đạp, 1 tay đua áo vàng quần đen, 1 tay đua áo xanh dương quần đen và 1 tay đua áo xanh dương quần đỏ",
    "search_mode": "hybrid",
    "top_k": 25
}

response = requests.post(url, json=payload)
if response.status_code == 200:
    results = response.json()
    for i, res in enumerate(results):
        print(f"Rank {i+1}: {res['video_id']} - Frame {res['frame_id']} - Score {res.get('score', 0)}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
