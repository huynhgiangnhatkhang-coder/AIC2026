import sys
try:
    from pymilvus import MilvusClient
except ImportError:
    print("Vui lòng cài đặt pymilvus: pip install pymilvus")
    sys.exit(1)

def inspect_database():
    uri = "http://aicpc.sytes.net:19530"
    collection_name = "AIC26_Frames"
    
    print(f"Đang kết nối tới Milvus tại: {uri}")
    try:
        client = MilvusClient(uri=uri)
        collections = client.list_collections()
        print(f"Các Collections hiện có: {collections}")
        
        if collection_name in collections:
            print(f"\n--- Đang kiểm tra Collection '{collection_name}' ---")
            
            # Kiểm tra Schema (Cấu trúc dữ liệu)
            desc = client.describe_collection(collection_name)
            print("\n1. Cấu trúc (Schema):")
            for field in desc["fields"]:
                print(f"  - Field: {field['name']} | Type: {field['type']} | Is Primary: {field.get('is_primary', False)}")
                if field['name'] == 'embedding' or field['name'] == 'vector':
                    print(f"    -> Kích thước (Dimension): {field['params'].get('dim', 'N/A')}")
            
            # Kiểm tra số lượng Record
            stats = client.get_collection_stats(collection_name)
            print(f"\n2. Số lượng dữ liệu (Row count): {stats.get('row_count', 'Unknown')}")
            
            # Thử lấy ra 1 dòng dữ liệu mẫu
            print("\n3. Dữ liệu mẫu (Sample Data):")
            try:
                # Query 1 dòng ngẫu nhiên (không lấy embedding để khỏi in ra màn hình quá dài)
                sample = client.query(
                    collection_name=collection_name,
                    filter="frame_id >= 0", 
                    output_fields=["video_id", "frame_id"], # Chỉ lấy các trường cơ bản
                    limit=1
                )
                if sample:
                    print(sample[0])
                else:
                    print("Không lấy được dữ liệu mẫu.")
            except Exception as e:
                print(f"Không thể query dữ liệu mẫu: {e}")
                
        else:
            print(f"Không tìm thấy Collection '{collection_name}'!")
            
    except Exception as e:
        print(f"Lỗi kết nối: {e}")

if __name__ == "__main__":
    inspect_database()
