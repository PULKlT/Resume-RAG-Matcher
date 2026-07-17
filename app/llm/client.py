"""
Shared LLM call wrapper (Ollama default / Groq alt), extracted here since
hyde.py, query_expansion.py, and synthesizer.py all need it.

NOTE: update app/retrieval/hyde.py to `from app.llm.client import call_llm`
and delete its local copy of this function.
"""
import requests

from app.config import LLM_PROVIDER, OLLAMA_BASE_URL, OLLAMA_MODEL, GROQ_API_KEY, GROQ_MODEL


def call_llm(prompt: str, max_tokens: int = 300, temperature: float = 0.7) -> str:
    if LLM_PROVIDER == "ollama":
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                  "options": {"num_predict": max_tokens, "temperature": temperature}},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()

    elif LLM_PROVIDER == "groq":
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return completion.choices[0].message.content.strip()

    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")


if __name__ == "__main__":
    print(call_llm("Say hello in 5 words.", max_tokens=20))
