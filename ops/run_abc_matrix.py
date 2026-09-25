import os
import sys
import json
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from infrastructure.llm_adapter import AsyncOpenRouterAdapter, _load_env_if_present
from ops.run_abc_clinical_eval_v2 import evaluate_baseline_turn, evaluate_generic_harness_turn, evaluate_kernel_turn

async def run_cheap_model():
    _load_env_if_present()
    print("===========================================================")
    print("STARTING A/B/C MATRIX RUNNER - GPT-4o-Mini")
    print("===========================================================")
    
    cases = []
    with open("evals/mtsamples/pmc_adversarial_15.jsonl", "r") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
                
    or_key = os.getenv("OPENROUTER_API_KEY")
    llm = AsyncOpenRouterAdapter(api_key=or_key, model="openai/gpt-4o-mini")
    
    results = {"A": [], "B": [], "C": []}
    
    for idx, case in enumerate(cases):
        print(f"Running Case {idx+1}/{len(cases)} [{case['id']}]...")
        
        # Track A
        res_a = await evaluate_baseline_turn(llm, case)
        results["A"].append(res_a)
        
        # Track B
        res_b = await evaluate_generic_harness_turn(llm, case)
        results["B"].append(res_b)
        
        # Track C
        res_c = await evaluate_kernel_turn(llm, case, str(idx))
        results["C"].append(res_c)
        
    # Process results for Obsidian
    with open("reports/gpt4o_mini_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("DONE! Results saved to reports/gpt4o_mini_results.json")

if __name__ == "__main__":
    asyncio.run(run_cheap_model())
