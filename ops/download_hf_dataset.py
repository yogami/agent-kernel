import json
import os

try:
    from datasets import load_dataset
except ImportError:
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets", "pandas", "huggingface_hub"])
    from datasets import load_dataset

def main():
    print("Downloading MTSamples dataset from HuggingFace...")
    # Load dataset
    dataset = load_dataset("tizfa/mtsamples", split="train")
    
    os.makedirs("evals/mtsamples", exist_ok=True)
    out_path = "evals/mtsamples/mtsamples.jsonl"
    
    with open(out_path, "w", encoding="utf-8") as f:
        count = 0
        for row in dataset:
            if "text" in row and row["text"]:
                obj = {
                    "id": f"mtsamples_{count}",
                    "specialty": row.get("medical_specialty", "unknown").strip(),
                    "text": row["text"]
                }
                f.write(json.dumps(obj) + "\n")
                count += 1
                
    print(f"Successfully processed {count} clinical notes into {out_path}")

if __name__ == "__main__":
    main()
