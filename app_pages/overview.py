import streamlit as st
import pandas as pd
from core.config import MODELS_INFO, MODEL_A, MODEL_B
from core.storage import get_kpi_summary

st.title("Сравнение языковых моделей и времени инференса")
st.caption("Лабораторная работа №2 • Локальный инференс на базе Ollama и NVIDIA GTX 1650")

# 1. Верхнеуровневые KPI-метрики
st.subheader("Сводные показатели эксперимента", anchor=False)

kpis = get_kpi_summary()

col1, col2, col3, col4 = st.columns(4)

with col1:
    with st.container(border=True):
        st.markdown("**TTFT** *(Prefill)*")
        st.caption("Задержка до первого токена")
        st.markdown(f"**Llama:** `{kpis['mean_ttft_a']} с`  \n**Qwen:** `{kpis['mean_ttft_b']} с`")
        st.caption("0.02 с в пользу Llama")

with col2:
    with st.container(border=True):
        st.markdown("**TBT** *(Decode)*")
        st.caption("Время между токенами")
        st.markdown(f"**Llama:** `{kpis['mean_tbt_a']} мс`  \n**Qwen:** `{kpis['mean_tbt_b']} мс`")
        st.caption("Скорость: ~49–50 токенов/с")

with col3:
    with st.container(border=True):
        st.markdown("**Throughput**")
        st.caption("Итоговая пропускная способность")
        st.markdown(f"**Llama:** `{kpis['mean_throughput_a']} т/с`  \n**Qwen:** `{kpis['mean_throughput_b']} т/с`")
        st.caption("+4.4 т/с в пользу Qwen")

with col4:
    with st.container(border=True):
        st.markdown("**LLM-as-a-Judge**")
        st.caption("Слепое сравнение ответов")
        st.markdown(f"**Qwen:** `8 побед (80%)`  \n**Llama:** `1 победа (10%)`")
        st.caption(f"Ср. балл: {kpis['judge_mean_b']} против {kpis['judge_mean_a']} из 10")

st.divider()

# 2. Аппаратный стенд и конфигурация развёртывания
col_env, col_models = st.columns([1, 1])

with col_env:
    with st.container(border=True):
        st.subheader("Аппаратно-программный стенд", anchor=False)
        st.markdown(
            """
            * **Графический ускоритель (GPU):** NVIDIA GeForce GTX 1650 (4 ГБ GDDR6 VRAM, Turing).
            * **Загрузка в VRAM:** 100% слоев модели размещены в видеопамяти GPU (без сброса в RAM).
            * **Среда выполнения инференса:** Локальный сервер **Ollama** v0.x с поддержкой CUDA.
            * **Интерфейс обращения:** OpenAI-совместимый REST API (`http://localhost:11434/v1`).
            * **Режим получения ответов:** Потоковый (`stream=True`) для точного замера фаз prefill и decode.
            """
        )

with col_models:
    with st.container(border=True):
        st.subheader("Методология исследования", anchor=False)
        st.markdown(
            """
            1. **Замер инференса:** 10 разнородных задач (QA, код, логика, суммаризация, перевод, JSON).
            2. **Метрики скорости:** TTFT (Time to First Token), TBT/TPOT (Time Between Tokens), Latency (U2E), Throughput.
            3. **Автоматическое качество:** BLEU и ROUGE-1/L относительно эталонов с учетом морфологии русского языка.
            4. **Экспертная оценка:** Слепое попарное судейство (LLM-as-a-Judge) с обоснованием арбитра.
            """
        )

st.divider()

# 3. Спецификация сравниваемых моделей
st.subheader("Спецификации сравниваемых моделей", anchor=False)

spec_data = []
for model_id, info in MODELS_INFO.items():
    spec_data.append({
        "Модель": info.name,
        "Разработчик": info.developer,
        "Число параметров": info.parameters,
        "Архитектура": info.architecture,
        "Контекстное окно": info.context_window,
        "Квантование": info.quantization,
        "Потребление VRAM": info.vram_usage
    })

st.dataframe(
    pd.DataFrame(spec_data),
    width="stretch",
    hide_index=True
)
