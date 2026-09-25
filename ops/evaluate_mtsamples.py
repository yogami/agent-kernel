import asyncio
import json
import os
import sys
from pathlib import Path

from core.execution_loop import ExecutionEngine
from domain.models import ConfigPack, StreamEvent, StreamEventType
from domain.state_machine import KernelState
from infrastructure.llm_adapter import AsyncOpenRouterAdapter, MockLLMAdapter
from infrastructure.sqlite_episode_store import SQLiteEpisodeStore
from infrastructure.sqlite_fact_store import SQLiteFactStore
from core.context_ram import ContextRAM
from tools.registry import ToolRegistry
from tools.clinical_tools import DeidentifyTextTool, MedicalOntologyMapperTool, ClinicalAssertionCheckerTool
from core.causal_gate import CausalPreFlightGate

async def run_evaluation(num_samples: int = 10):
    dataset_path = Path("evals/mtsamples/mtsamples.jsonl")
    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}")
        return

    # Load samples
    samples = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
            if len(samples) >= num_samples:
                break
                
    print(f"Loaded {len(samples)} samples from MTSamples.")

    from infrastructure.llm_adapter import create_default_llm_adapter
    llm = create_default_llm_adapter()
    print(f"Active LLM Adapter: {llm.__class__.__name__}")
    if isinstance(llm, MockLLMAdapter):
        # Seed mock responses for extraction
        for s in samples:
            llm.set_scripted_response(s["text"][:50], {
                "content": "",
                "tool_calls": [{"id": "1", "type": "function", "function": {"name": "deidentify_text", "arguments": json.dumps({"text": s["text"]})}}]
            })

    store = SQLiteEpisodeStore(":memory:")
    fact_store = SQLiteFactStore(":memory:")
    context_ram = ContextRAM(fact_store)
    
    registry = ToolRegistry()
    registry.register(DeidentifyTextTool())
    registry.register(MedicalOntologyMapperTool())
    registry.register(ClinicalAssertionCheckerTool())
    
    engine = ExecutionEngine(llm, store, registry, context_ram)
    
    config = ConfigPack(
        version="1.0",
        system_prompt="You are a clinical agent. You must de-identify the provided text using the 'deidentify_text' tool, and extract clinical facts."
    )
    
    success_count = 0
    total_cost = 0.0
    
    for idx, sample in enumerate(samples):
        print(f"\n--- Evaluating Sample {idx+1}/{len(samples)} [{sample['specialty']}] ---")
        prompt = f"Please process the following clinical note:\n\n{sample['text']}"
        session_id = f"eval_sess_{idx}"
        
        if isinstance(llm, MockLLMAdapter):
            prompt = sample["text"][:50]
            
        print(f"Running agent stream...")
        try:
            tool_calls_made = 0
            final_content = ""
            
            async for event in engine.run_turn_stream(prompt, session_id, config):
                if event.event == StreamEventType.TOOL_CALL:
                    tool_name = event.payload.get("tool_name", "")
                    print(f"  🔧 Agent called tool: {tool_name}")
                    tool_calls_made += 1
                elif event.event == StreamEventType.TOKEN:
                    final_content += event.payload.get("chunk", "")
                elif event.event == StreamEventType.TURN_COMPLETED:
                    total_cost += event.payload.get("cost_usd", 0.0)
                    
            print(f"  ✅ Execution completed. Tools used: {tool_calls_made}")
            success_count += 1
            
        except Exception as e:
            print(f"  ❌ Execution failed: {e}")

        if idx < len(samples) - 1 and not isinstance(llm, MockLLMAdapter):
            await asyncio.sleep(4)
            
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Samples Evaluated:  {len(samples)}")
    print(f"Successful Runs:    {success_count} / {len(samples)}")
    print(f"Total Cost (USD):   ${total_cost:.4f}")
    
    os.makedirs("reports", exist_ok=True)
    with open("reports/mtsamples_eval_results.json", "w") as f:
        json.dump({
            "samples_evaluated": len(samples),
            "successful_runs": success_count,
            "total_cost_usd": total_cost,
            "dataset": "mtsamples"
        }, f, indent=2)
    print("Saved report to reports/mtsamples_eval_results.json")

if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    asyncio.run(run_evaluation(count))
