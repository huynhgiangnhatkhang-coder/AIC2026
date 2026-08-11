"""
Script kiem tra end-to-end (ASCII-safe, khong dung emoji):
1. Import check
2. Milvus database check
3. Cau truc thu muc Dataset
4. Config.yaml check
"""
import sys
import os

errors = []
warnings = []

SEP = "=" * 60

print(SEP)
print("  AIC2026 Baseline - End-to-End Check")
print(SEP)

# ─── 1. IMPORT CHECK ───────────────────────────────────────
print("\n[1/5] Import check...")

try:
    import torch
    cuda_ok = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "N/A"
    print(f"  [OK] torch {torch.__version__} | CUDA={cuda_ok} | GPU={gpu_name}")
except ImportError as e:
    errors.append(f"torch: {e}")
    print(f"  [ERR] torch: {e}")

try:
    import clip
    print(f"  [OK] clip (openai/clip)")
except ImportError:
    try:
        import open_clip
        print(f"  [OK] open_clip {open_clip.__version__} (fallback OK)")
    except ImportError as e:
        errors.append(f"clip/open_clip: {e}")
        print(f"  [ERR] clip: {e}")

try:
    from pymilvus import MilvusClient
    print(f"  [OK] pymilvus")
except ImportError as e:
    errors.append(f"pymilvus: {e}")
    print(f"  [ERR] pymilvus: {e}")

try:
    from transformers import MarianMTModel, MarianTokenizer
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    import transformers
    print(f"  [OK] transformers {transformers.__version__}")
except ImportError as e:
    errors.append(f"transformers: {e}")
    print(f"  [ERR] transformers: {e}")

try:
    import sentencepiece
    print(f"  [OK] sentencepiece")
except ImportError as e:
    errors.append(f"sentencepiece: {e}")
    print(f"  [ERR] sentencepiece: {e}")

try:
    import sacremoses
    print(f"  [OK] sacremoses")
except ImportError as e:
    errors.append(f"sacremoses: {e}")
    print(f"  [ERR] sacremoses: {e}")

try:
    import numpy as np
    print(f"  [OK] numpy {np.__version__}")
except ImportError as e:
    errors.append(f"numpy: {e}")
    print(f"  [ERR] numpy: {e}")

try:
    import yaml
    print(f"  [OK] PyYAML")
except ImportError as e:
    errors.append(f"yaml: {e}")
    print(f"  [ERR] PyYAML: {e}")
    sys.exit(1)  # can't continue without yaml

try:
    from PIL import Image
    print(f"  [OK] Pillow")
except ImportError as e:
    errors.append(f"Pillow: {e}")
    print(f"  [ERR] Pillow: {e}")

# ─── 2. CONFIG CHECK ───────────────────────────────────────
print("\n[2/5] Config check (config.yaml)...")
cfg = None
try:
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print(f"  [OK] config.yaml loaded")
    print(f"       retrieval_backend : {cfg.get('retrieval_backend', 'MISSING')}")
    print(f"       milvus_db_path    : {cfg['index']['milvus_db_path']}")
    print(f"       keyframes_root    : {cfg['data'].get('keyframes_root', '*** MISSING ***')}")
    print(f"       npy_dirs[0]       : {cfg['data'].get('npy_dirs', ['MISSING'])[0]}")
    dino_cfg = cfg.get('dino', {})
    print(f"       dino.alpha_clip   : {dino_cfg.get('alpha_clip', '*** MISSING ***')}")
    print(f"       dino.top_k_coarse : {dino_cfg.get('top_k_coarse', '*** MISSING ***')}")
except Exception as e:
    errors.append(f"config.yaml: {e}")
    print(f"  [ERR] config.yaml: {e}")

# ─── 3. MILVUS DATABASE CHECK ──────────────────────────────
print("\n[3/5] Milvus database check...")
if cfg:
    db_path = cfg['index']['milvus_db_path']
    col_name = cfg['index']['milvus_collection']
    if os.path.exists(db_path):
        try:
            client = MilvusClient(db_path)
            client.load_collection(col_name)
            stats = client.get_collection_stats(col_name)
            row_count = stats.get('row_count', '?')
            print(f"  [OK] Milvus DB: {db_path}")
            print(f"       Collection : {col_name} | {row_count} records")
        except Exception as e:
            errors.append(f"Milvus connect error: {e}")
            print(f"  [ERR] Milvus connect: {e}")
    else:
        errors.append(f"Milvus DB not found: {db_path}")
        print(f"  [ERR] Milvus DB not found: {db_path}")
        print(f"       Fix: python scripts/02b_build_milvus_db.py --config config.yaml")

# ─── 4. DIRECTORY CHECK ────────────────────────────────────
print("\n[4/5] Directory structure check...")
if cfg:
    checks = {
        "keyframes_root": cfg['data'].get('keyframes_root', ''),
        "npy_dir"       : cfg['data'].get('npy_dirs', [''])[0],
        "milvus_db"     : cfg['index']['milvus_db_path'],
        "submissions"   : cfg['submission']['output_dir'],
    }

    for name, path in checks.items():
        if not path:
            warnings.append(f"{name}: empty path in config")
            print(f"  [WARN] {name}: no path configured")
            continue

        if os.path.exists(path):
            if os.path.isdir(path):
                contents = os.listdir(path)
                print(f"  [OK] {name}: {path}  ({len(contents)} items)")
            else:
                print(f"  [OK] {name}: {path}  (file exists)")
        else:
            if name == "submissions":
                warnings.append(f"{name}: will be created on first run")
                print(f"  [WARN] {name}: {path} (not yet created, will auto-create)")
            else:
                errors.append(f"{name}: NOT FOUND -> {path}")
                print(f"  [ERR] {name}: NOT FOUND -> {path}")

    # Kiem tra cau truc ben trong keyframes_root
    kf_root = cfg['data'].get('keyframes_root', '')
    if kf_root and os.path.exists(kf_root):
        print(f"\n       Detail - keyframes_root contents:")
        try:
            all_items = os.listdir(kf_root)
            batches = sorted([d for d in all_items if d.startswith("Keyframes_")])
            other   = [d for d in all_items if not d.startswith("Keyframes_")]

            if batches:
                for b in batches[:8]:
                    b_kf = os.path.join(kf_root, b, "keyframes")
                    if os.path.exists(b_kf):
                        n_vid = len(os.listdir(b_kf))
                        print(f"         [OK] {b}/keyframes/  ->  {n_vid} videos")
                    else:
                        # Thu xem co thu muc con nao khong
                        b_path = os.path.join(kf_root, b)
                        sub = os.listdir(b_path) if os.path.exists(b_path) else []
                        print(f"         [ERR] {b}/keyframes/  NOT FOUND  (found: {sub})")
                        errors.append(f"{b}: missing 'keyframes' subfolder")
                if len(batches) > 8:
                    print(f"         ... ({len(batches)} Keyframes_* batches total)")
            else:
                print(f"         [ERR] No 'Keyframes_L*' folders found!")
                print(f"         Items found: {all_items[:10]}")
                errors.append("keyframes_root: no Keyframes_Lxx folders found")

            if other:
                print(f"         Other items: {other[:5]}")
        except Exception as e:
            print(f"         [ERR] Could not list dir: {e}")

# ─── 5. NPY FILES CHECK ────────────────────────────────────
print("\n[5/5] NPY vector files check...")
if cfg:
    for npy_dir in cfg['data'].get('npy_dirs', []):
        if os.path.exists(npy_dir):
            npy_files = sorted([f for f in os.listdir(npy_dir) if f.endswith('.npy')])
            print(f"  [OK] {npy_dir}")
            print(f"       {len(npy_files)} .npy files found")
            if npy_files:
                print(f"       First: {npy_files[0]}  |  Last: {npy_files[-1]}")
        else:
            errors.append(f"npy_dir not found: {npy_dir}")
            print(f"  [ERR] NOT FOUND: {npy_dir}")

# ─── SUMMARY ───────────────────────────────────────────────
print("\n" + SEP)
print("  SUMMARY")
print(SEP)
if not errors and not warnings:
    print("  ALL OK - Ready to run search!")
else:
    if warnings:
        print(f"\n  WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"    - {w}")
    if errors:
        print(f"\n  ERRORS TO FIX ({len(errors)}):")
        for e in errors:
            print(f"    - {e}")
        sys.exit(1)
    else:
        print("\n  No blocking errors. Warnings noted above.")
