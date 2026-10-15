"""
Глобальные конфигурационные параметры для системы бенчмаркинга.
"""
from dataclasses import dataclass

OLLAMA_BASE_URL: str = "http://localhost:11434/v1"
OLLAMA_API_KEY: str = "ollama"

MODEL_A: str = "llama3.2:3b"
MODEL_B: str = "qwen2.5:3b"
JUDGE_MODEL: str = "qwen2.5:3b"

RESULTS_FILE: str = "data/benchmark_results.json"

@dataclass(frozen=True)
class ModelMeta:
    name: str
    developer: str
    parameters: str
    architecture: str
    context_window: str
    quantization: str
    vram_usage: str

MODELS_INFO = {
    MODEL_A: ModelMeta(
        name="Llama 3.2 3B Instruct",
        developer="Meta AI",
        parameters="3.21 млрд",
        architecture="Decoder-only (Dense Transformer, GQA)",
        context_window="128K токенов",
        quantization="Q4_K_M (4-bit)",
        vram_usage="~2.0 ГБ"
    ),
    MODEL_B: ModelMeta(
        name="Qwen 2.5 3B Instruct",
        developer="Alibaba Cloud",
        parameters="3.09 млрд",
        architecture="Decoder-only (Dense Transformer, GQA, RoPE)",
        context_window="128K токенов",
        quantization="Q4_K_M (4-bit)",
        vram_usage="~1.9 ГБ"
    )
}
