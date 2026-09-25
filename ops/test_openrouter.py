import os
import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from infrastructure.llm_adapter import AsyncOpenRouterAdapter, _load_env_if_present

async def main():
    _load_env_if_present()
    or_key = os.getenv("OPENROUTER_API_KEY")
    if not or_key:
        print("No OPENROUTER_API_KEY found")
        return
    print(f"Key starts with: {or_key[:10]}...")
    llm = AsyncOpenRouterAdapter(api_key=or_key, model="openai/gpt-4o-mini")
    
    messages = [{"role": "user", "content": "Hello!"}]
    try:
        resp = await llm.generate_async(messages)
        print("Response keys:", resp.keys())
        if 'error' in resp:
            print("ERROR:", resp['error'])
        elif 'choices' in resp:
            print("SUCCESS:", resp['choices'][0]['message']['content'])
        else:
            print("RAW RESPONSE:", resp)
    except Exception as e:
        print(f"Exception: {e}")

asyncio.run(main())
