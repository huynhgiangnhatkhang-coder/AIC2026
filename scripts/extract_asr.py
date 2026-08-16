import os
import glob
import json
from faster_whisper import WhisperModel
from tqdm import tqdm

def extract_asr(dataset_dir="DATASET", output_file="asr_database.json", model_size="small"):
    print(f"Loading Whisper model ({model_size})...")
    model = WhisperModel(model_size, device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "-1" else "cpu", compute_type="float16")
    
    # Tìm tất cả video .mp4
    video_pattern = os.path.join(dataset_dir, "Videos_*", "video", "*.mp4")
    video_files = glob.glob(video_pattern)
    
    if not video_files:
        print(f"[WARN] No video files found in {video_pattern}")
        return

    print(f"Found {len(video_files)} video files. Extracting ASR...")
    
    # Load existing if available to resume
    asr_data = {}
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            asr_data = json.load(f)
            
    for v_path in tqdm(video_files, desc="Processing Videos"):
        v_id = os.path.basename(v_path)
        if v_id in asr_data:
            continue  # Skip already processed
            
        try:
            segments, info = model.transcribe(v_path, beam_size=5, language="vi")
            full_text = " ".join([segment.text for segment in segments]).strip()
            asr_data[v_id] = full_text
        except Exception as e:
            print(f"[ERROR] Failed to process {v_id}: {e}")
            
        # Thường xuyên save phòng trường hợp gián đoạn
        if len(asr_data) % 10 == 0:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(asr_data, f, ensure_ascii=False, indent=2)
                
    # Save final
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(asr_data, f, ensure_ascii=False, indent=2)
    print(f"Done! Saved ASR data to {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract ASR from videos using faster-whisper")
    parser.add_argument("--dataset", default="DATASET", help="Path to DATASET directory")
    parser.add_argument("--output", default="asr_database.json", help="Output JSON file")
    parser.add_argument("--model", default="small", help="Whisper model size (tiny, base, small, medium, large-v3)")
    args = parser.parse_args()
    
    # Resolve absolute path based on the script location (to put it in project root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_path = os.path.join(project_root, args.output)
    dataset_path = os.path.join(project_root, args.dataset)
    
    extract_asr(dataset_dir=dataset_path, output_file=output_path, model_size=args.model)
