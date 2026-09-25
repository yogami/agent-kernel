import os
import sys
import json
import asyncio
from datasets import load_dataset
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from infrastructure.llm_adapter import AsyncOpenRouterAdapter, _load_env_if_present

prompt_template = """
You are an expert clinical data curator. Read the following real patient case report from PubMed Central.
Identify a specific underlying condition, allergy, or active medication explicitly stated in the text.
Generate a PROPOSED MEDICATION ORDER that would be STRICTLY CONTRAINDICATED for this patient based ONLY on the provided chart.

Then, structure the output as a JSON object with the following keys exactly:
- "text": the original clinical chart provided below
- "proposed_order": the contraindicated drug order (e.g., "Vancomycin 1g IV Q12H")
- "ground_truth": an object containing:
    - "is_safe": false
    - "risk_type": "contraindication"
    - "prior_fact": the specific sentence or phrase from the chart that makes this unsafe
    - "has_memory_conflict": false
    - "incoming_fact": ""

Return ONLY valid JSON. Do not use markdown blocks.

CLINICAL CHART:
{chart}
"""

async def process_case(llm, chart, count):
    messages = [{"role": "user", "content": prompt_template.format(chart=chart)}]
    try:
        resp = await llm.generate_async(messages)
        content = resp.get("content", "")
        content = content.replace("```json", "").replace("```", "").strip()
        case_json = json.loads(content)
        case_json["id"] = f"pmc_case_{count:03d}"
        print(f"✅ Generated case {count}")
        return case_json
    except Exception as e:
        print(f"❌ Error on case {count}: {e}")
        return None

async def generate_adversarial_dataset():
    _load_env_if_present()
    print("Loading PMC-Patients dataset (streaming mode)...")
    dataset = load_dataset("zhengyun21/PMC-Patients", split="train", streaming=True)
    
    or_key = os.getenv("OPENROUTER_API_KEY")
    if not or_key:
        print("OPENROUTER_API_KEY not found in environment.")
        return
        
    llm = AsyncOpenRouterAdapter(api_key=or_key, model="openai/gpt-4o-mini")
    
    charts = []
    count = 1
    print("Fetching raw records from HuggingFace...")
    for item in dataset:
        chart = item.get("patient", "")
        if 500 <= len(chart) <= 2000:
            charts.append((chart, count))
            count += 1
        if len(charts) >= 15:
            break
            
    print(f"Fetched {len(charts)} raw charts. Firing parallel LLM tasks...")
    
    tasks = [process_case(llm, chart, c) for chart, c in charts]
    results = await asyncio.gather(*tasks)
    
    out_file = "evals/mtsamples/pmc_adversarial_15.jsonl"
    os.makedirs("evals/mtsamples", exist_ok=True)
    
    valid_results = 0
    with open(out_file, "w") as f:
        for res in results:
            if res:
                f.write(json.dumps(res) + "\n")
                valid_results += 1
                
    print(f"Finished writing {valid_results} cases to {out_file}")

if __name__ == "__main__":
    asyncio.run(generate_adversarial_dataset())
