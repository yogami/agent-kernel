import json
import os
from huggingface_hub import HfApi
from datasets import load_dataset

def main():
    api = HfApi()
    datasets = list(api.list_datasets(search="mtsamples", limit=5))
    if not datasets:
        print("No datasets found!")
        return
    dataset_name = datasets[0].id
    print(f"Found best MTSamples dataset: {dataset_name}")
    
    print("Downloading dataset...")
    dataset = load_dataset(dataset_name, split="train")
    
    os.makedirs("evals/mtsamples", exist_ok=True)
    out_path = "evals/mtsamples/mtsamples.jsonl"
    
    with open(out_path, "w", encoding="utf-8") as f:
        count = 0
        for row in dataset:
            text = row.get("text") or row.get("transcription") or str(row)
            specialty = row.get("medical_specialty") or "unknown"
            if isinstance(specialty, str):
                specialty = specialty.strip()
            
            obj = {
                "id": f"mtsamples_{count}",
                "specialty": specialty,
                "text": text
            }
            f.write(json.dumps(obj) + "\n")
            count += 1
                
    print(f"Successfully processed {count} clinical notes into {out_path}")

if __name__ == "__main__":
    main()
