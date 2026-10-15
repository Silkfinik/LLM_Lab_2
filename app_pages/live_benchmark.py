import streamlit as st
import time
from core.config import MODEL_A, MODEL_B
from core.engine import stream_with_metrics
from core.dataset import PROMPTS_DATA

st.title("Интерактивный полигон инференса")
st.caption("Тестирование моделей в реальном времени с замером аппаратных метрик на вашей GTX 1650")

# 1. Выбор готового пресета или ввод своего промпта
prompt_preset_names = ["-- Пользовательский запрос --"] + [f"№{p['id']}: {p['category']}" for p in PROMPTS_DATA]

selected_preset = st.selectbox(
    "Выберите пример запроса или введите свой:",
    options=prompt_preset_names,
    index=0
)

default_text = ""
if selected_preset != "-- Пользовательский запрос --":
    p_id = int(selected_preset.split(":")[0].replace("№", ""))
    found = next((p for p in PROMPTS_DATA if p["id"] == p_id), None)
    if found:
        default_text = found["prompt"]

user_prompt = st.text_area(
    "Текст запроса для инференса:",
    value=default_text,
    height=100,
    placeholder="Введите любой вопрос на русском или английском языке..."
)

col_btn, _ = st.columns([1, 3])
with col_btn:
    start_btn = st.button("Запустить инференс", type="primary", disabled=not bool(user_prompt.strip()))

if start_btn and user_prompt.strip():
    st.divider()
    col1, col2 = st.columns(2)

    # 1. Запуск Модели А
    with col1:
        st.subheader("Llama 3.2 3B", anchor=False)
        with st.container(border=True):
            status_a = st.empty()
            status_a.caption("Генерация ответа (стриминг)...")
            
            stream_gen_a, metrics_a = stream_with_metrics(MODEL_A, user_prompt)
            st.write_stream(stream_gen_a)
            status_a.empty()

        # Вывод метрик А
        with st.container(border=True):
            m1, m2 = st.columns(2)
            m1.metric("TTFT (prefill)", f"{metrics_a['ttft_sec']} с")
            m2.metric("TBT (decode)", f"{metrics_a['tbt_ms']} мс")
            
            m3, m4 = st.columns(2)
            m3.metric("Throughput", f"{metrics_a['throughput_tok_s']} т/с")
            m4.metric("Токенов / Время", f"{metrics_a['tokens']} т / {metrics_a['total_latency_sec']} с")

    # 2. Запуск Модели Б
    with col2:
        st.subheader("Qwen 2.5 3B", anchor=False)
        with st.container(border=True):
            status_b = st.empty()
            status_b.caption("Генерация ответа (стриминг)...")
            
            stream_gen_b, metrics_b = stream_with_metrics(MODEL_B, user_prompt)
            st.write_stream(stream_gen_b)
            status_b.empty()

        # Вывод метрик Б
        with st.container(border=True):
            mb1, mb2 = st.columns(2)
            mb1.metric("TTFT (prefill)", f"{metrics_b['ttft_sec']} с")
            mb2.metric("TBT (decode)", f"{metrics_b['tbt_ms']} мс")
            
            mb3, mb4 = st.columns(2)
            mb3.metric("Throughput", f"{metrics_b['throughput_tok_s']} т/с")
            mb4.metric("Токенов / Время", f"{metrics_b['tokens']} т / {metrics_b['total_latency_sec']} с")
