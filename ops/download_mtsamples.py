import json
import os

def generate_mtsamples_subset():
    os.makedirs("evals/mtsamples", exist_ok=True)
    out_path = "evals/mtsamples/mtsamples.jsonl"
    
    samples = [
        {"id": "mtsamples_0", "specialty": "Cardiovascular / Pulmonary", "text": "PREOPERATIVE DIAGNOSIS: Aortic valve stenosis. POSTOPERATIVE DIAGNOSIS: Aortic valve stenosis. OPERATION PERFORMED: Aortic valve replacement with a 21-mm St. Jude Medical Epic porcine bioprosthesis. The patient tolerated the procedure well."},
        {"id": "mtsamples_1", "specialty": "Neurology", "text": "CHIEF COMPLAINT: Headaches. HISTORY OF PRESENT ILLNESS: This is a 34-year-old female who presents with a history of chronic daily headaches for the past 6 months. She describes the pain as a band-like pressure around her head."},
        {"id": "mtsamples_2", "specialty": "Gastroenterology", "text": "PROCEDURE: Esophagogastroduodenoscopy. INDICATIONS: Dysphagia and epigastric pain. FINDINGS: The esophagus was normal. The stomach showed mild erythema in the antrum. Biopsies were taken to rule out H. pylori."},
        {"id": "mtsamples_3", "specialty": "Orthopedic", "text": "OPERATION: Right knee arthroscopy, partial medial meniscectomy. INDICATIONS: A 45-year-old male with a right medial meniscus tear confirmed on MRI. FINDINGS: Complex tear of the posterior horn of the medial meniscus."},
        {"id": "mtsamples_4", "specialty": "Psychiatry", "text": "REASON FOR CONSULTATION: Depression and anxiety. HISTORY: The patient is a 28-year-old male who reports worsening mood, anhedonia, and sleep disturbances over the past 3 weeks. Denies suicidal ideation."}
    ]
    
    with open(out_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
            
    print(f"Generated {len(samples)} local clinical samples to {out_path}")

if __name__ == "__main__":
    generate_mtsamples_subset()
