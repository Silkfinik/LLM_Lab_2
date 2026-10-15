"""
Движок инференса для измерения аппаратных метрик времени (TTFT, TBT, Latency, Throughput).
"""
import time
from typing import Dict, Any, Generator, Tuple
from openai import OpenAI
from core.config import OLLAMA_BASE_URL, OLLAMA_API_KEY

def get_client() -> OpenAI:
    """Возвращает инициализированный OpenAI-совместимый клиент."""
    return OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)

def measure_inference(model_name: str, prompt: str) -> Dict[str, Any]:
    """
    Выполняет streaming-запрос к модели и замеряет:
    - TTFT (Time to First Token)
    - Total Latency (U2E)
    - TBT / TPOT (Time Between Tokens)
    - Throughput (токенов/сек)
    """
    client = get_client()
    t0 = time.perf_counter()

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        stream_options={"include_usage": True}
    )

    first_token_time = None
    chunks = []
    usage = None

    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            chunks.append(chunk.choices[0].delta.content)
        if hasattr(chunk, "usage") and chunk.usage:
            usage = chunk.usage

    t_end = time.perf_counter()
    full_text = "".join(chunks).strip()
    total_latency = t_end - t0

    if first_token_time is not None:
        ttft = first_token_time - t0
    else:
        ttft = total_latency

    if usage and usage.completion_tokens:
        tokens_generated = usage.completion_tokens
    else:
        tokens_generated = max(1, len(full_text.split()) * 4 // 3)

    if tokens_generated > 1:
        tbt = (total_latency - ttft) / (tokens_generated - 1)
    else:
        tbt = 0.0

    throughput = tokens_generated / total_latency if total_latency > 0 else 0.0

    return {
        "text": full_text,
        "ttft_sec": round(ttft, 4),
        "total_latency_sec": round(total_latency, 4),
        "tbt_ms": round(tbt * 1000, 2),
        "throughput_tok_s": round(throughput, 2),
        "tokens": tokens_generated
    }

def stream_with_metrics(model_name: str, prompt: str) -> Tuple[Generator[str, None, None], Dict[str, Any]]:
    """
    Генератор для живого стриминга в Streamlit с подсчетом метрик.
    """
    client = get_client()
    t0 = time.perf_counter()
    metrics = {
        "ttft_sec": 0.0,
        "total_latency_sec": 0.0,
        "tbt_ms": 0.0,
        "throughput_tok_s": 0.0,
        "tokens": 0
    }

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        stream_options={"include_usage": True}
    )

    def text_gen():
        nonlocal metrics
        first_token = True
        chunks = []
        usage = None

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                if first_token:
                    metrics["ttft_sec"] = round(time.perf_counter() - t0, 4)
                    first_token = False
                token_text = chunk.choices[0].delta.content
                chunks.append(token_text)
                yield token_text
            if hasattr(chunk, "usage") and chunk.usage:
                usage = chunk.usage

        t_end = time.perf_counter()
        metrics["total_latency_sec"] = round(t_end - t0, 4)
        if usage and usage.completion_tokens:
            metrics["tokens"] = usage.completion_tokens
        else:
            metrics["tokens"] = max(1, len("".join(chunks).split()) * 4 // 3)

        if metrics["tokens"] > 1:
            tbt = (metrics["total_latency_sec"] - metrics["ttft_sec"]) / (metrics["tokens"] - 1)
            metrics["tbt_ms"] = round(tbt * 1000, 2)

        if metrics["total_latency_sec"] > 0:
            metrics["throughput_tok_s"] = round(metrics["tokens"] / metrics["total_latency_sec"], 2)

    return text_gen(), metrics
