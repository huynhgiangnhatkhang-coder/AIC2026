from pymilvus import MilvusClient, DataType
import time
import sys

def migrate():
    local_db = "./aic_kis_database_siglip.db"
    remote_uri = "http://aicpc.sytes.net:19530"
    
    collection_local = "kis_keyframes_siglip"
    collection_remote = "aic_siglip_test"
    
    print("1. Kết nối tới Milvus Local...")
    client_local = MilvusClient(local_db)
    client_local.load_collection(collection_local)
    
    print("2. Kết nối tới Milvus Remote...")
    client_remote = MilvusClient(uri=remote_uri)
    
    if client_remote.has_collection(collection_remote):
        print(f"Xóa collection cũ trên remote: {collection_remote}")
        client_remote.drop_collection(collection_remote)
        
    print("3. Tạo Schema trên Remote (768 dims)...")
    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="video_id", datatype=DataType.VARCHAR, max_length=200)
    schema.add_field(field_name="frame_id", datatype=DataType.INT64)
    schema.add_field(field_name="frame_filename", datatype=DataType.VARCHAR, max_length=100)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=768)
    
    client_remote.create_collection(collection_name=collection_remote, schema=schema)
    
    print("4. Đang khởi tạo iterator...")
    stats = client_local.get_collection_stats(collection_local)
    total_rows = stats.get('row_count', 0)
    print(f"Tổng số records: {total_rows}")
    
    try:
        query_iterator = client_local.query_iterator(
            collection_name=collection_local,
            output_fields=["video_id", "frame_id", "frame_filename", "embedding"],
            batch_size=2000
        )
    except Exception as e:
        print(f"Lỗi tạo iterator: {e}")
        # Fallback manual iterator by offset
        print("Fallback to offset/limit query")
        inserted = 0
        limit = 5000
        for offset in range(0, total_rows, limit):
            res = client_local.query(
                collection_name=collection_local,
                filter="frame_id >= 0",
                output_fields=["video_id", "frame_id", "frame_filename", "embedding"],
                limit=limit,
                offset=offset
            )
            batch = []
            for r in res:
                batch.append({
                    "video_id": r["video_id"],
                    "frame_id": r["frame_id"],
                    "frame_filename": r.get("frame_filename", ""),
                    "embedding": r["embedding"]
                })
            client_remote.insert(collection_name=collection_remote, data=batch)
            inserted += len(batch)
            print(f"  Đã copy: {inserted}/{total_rows}")
        query_iterator = None
    
    if query_iterator:
        inserted = 0
        batch = []
        while True:
            res = query_iterator.next()
            if not res:
                break
            
            for r in res:
                batch.append({
                    "video_id": r["video_id"],
                    "frame_id": r["frame_id"],
                    "frame_filename": r.get("frame_filename", ""),
                    "embedding": r["embedding"]
                })
                
            if len(batch) >= 2000:
                client_remote.insert(collection_name=collection_remote, data=batch)
                inserted += len(batch)
                batch = []
                print(f"  Đã copy: {inserted}/{total_rows}")
                    
        if batch:
            client_remote.insert(collection_name=collection_remote, data=batch)
            inserted += len(batch)
            
        print(f"  Hoàn tất copy: {inserted} records.")
    
    print("5. Tạo Index HNSW trên Remote...")
    index_params = client_remote.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        metric_type="IP",
        index_type="HNSW",
        params={"M": 16, "efConstruction": 256}
    )
    client_remote.create_index(collection_name=collection_remote, index_params=index_params)
    
    print("6. Xong! Đã nạp collection lên Remote RAM.")
    client_remote.load_collection(collection_remote)
    
    stats_remote = client_remote.get_collection_stats(collection_remote)
    print(f"Tổng records trên Remote: {stats_remote.get('row_count')}")

if __name__ == "__main__":
    migrate()
