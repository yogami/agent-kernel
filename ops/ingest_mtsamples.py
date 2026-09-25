import csv
import json
import os
import sys

def ingest_csv(csv_path: str):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please download it from Kaggle.")
        sys.exit(1)
        
    os.makedirs("evals/mtsamples", exist_ok=True)
    out_path = "evals/mtsamples/mtsamples.jsonl"
    
    count = 0
    with open(csv_path, "r", encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as out:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("transcription"):
                obj = {
                    "id": row.get("Unnamed: 0", f"mtsamples_{count}"),
                    "specialty": row.get("medical_specialty", "unknown").strip(),
                    "text": row["transcription"]
                }
                out.write(json.dumps(obj) + "\n")
                count += 1
                
    print(f"Successfully processed {count} clinical notes into {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest_mtsamples.py /path/to/mtsamples.csv")
    else:
        ingest_csv(sys.argv[1])
