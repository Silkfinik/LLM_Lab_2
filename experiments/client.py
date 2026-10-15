"""Native Ollama API: exact counters, raw stream events, no token estimates."""
import json
import time
import requests


class Ollama:
    def __init__(self, url="http://localhost:11434", timeout=300):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.http = requests.Session()
        self.http.trust_env = False  # local inference must not go through an HTTP proxy

    def api(self, path, body=None):
        response = self.http.request("GET" if body is None else "POST", self.url + path,
                                     json=body, timeout=(10, self.timeout))
        response.raise_for_status()
        result = response.json()
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def unload(self, model):
        self.api("/api/generate", {"model": model, "keep_alive": 0, "stream": False})
        active = self.api("/api/ps").get("models", [])
        if any(model in (m.get("name"), m.get("model")) for m in active):
            raise RuntimeError(f"Model still loaded after unload: {model}")

    def chat(self, model, prompt, options, system=None, output_format=None):
        messages = ([{"role": "system", "content": system}] if system else [])
        messages.append({"role": "user", "content": prompt})
        payload = dict(model=model, messages=messages, options=options,
                       stream=True, keep_alive="10m")
        if output_format is not None:
            payload["format"] = output_format
        start = time.perf_counter()
        events, parts, thoughts, arrivals = [], [], [], []
        final = None
        first = last = None
        with self.http.post(self.url + "/api/chat", json=payload, stream=True,
                            timeout=(10, self.timeout)) as response:
            response.raise_for_status()
            # chunk_size=1 avoids a 512-byte client buffer distorting first-arrival latency.
            for line in response.iter_lines(chunk_size=1):
                if not line:
                    continue
                event = json.loads(line)
                now = time.perf_counter()
                if "error" in event:
                    raise RuntimeError(event["error"])
                events.append({"elapsed_sec": now - start, "data": event})
                message = event.get("message", {})
                if message.get("thinking"):
                    thoughts.append(message["thinking"])
                if message.get("content"):
                    first = first if first is not None else now
                    last = now
                    arrivals.append(now - start)
                    parts.append(message["content"])
                if event.get("done"):
                    final = event
        finished = time.perf_counter()
        if final is None:
            raise RuntimeError("Stream ended without done=true; incomplete response")
        count = final.get("eval_count")
        if not isinstance(count, int) or count < 1:
            raise RuntimeError("Ollama did not return a positive exact eval_count")
        duration = final.get("eval_duration", 0) / 1e9
        text = "".join(parts)
        ttft = first - start if first is not None else None
        return dict(text=text, thinking="".join(thoughts),
                    ttft_sec=ttft, last_content_sec=last - start if last else None,
                    total_latency_sec=finished - start,
                    client_tpot_ms=((last - first) * 1000 / (count - 1)
                                    if first is not None and count > 1 and not thoughts else None),
                    server_tpot_ms=duration * 1000 / count if duration > 0 else None,
                    throughput_tok_s=count / (finished - start),
                    server_decode_tok_s=count / duration if duration > 0 else None,
                    chars_per_sec=len(text) / (finished - start), tokens=count,
                    token_count_source="ollama.eval_count", content_chunks=len(arrivals),
                    prompt_tokens=final.get("prompt_eval_count"),
                    prompt_cached_tokens=final.get("prompt_eval_cached_count"),
                    prompt_eval_sec=final.get("prompt_eval_duration", 0) / 1e9,
                    load_sec=final.get("load_duration", 0) / 1e9,
                    server_total_sec=final.get("total_duration", 0) / 1e9,
                    done_reason=final.get("done_reason"),
                    truncated=final.get("done_reason") == "length",
                    raw_final=final, stream_events=events, request=payload)
