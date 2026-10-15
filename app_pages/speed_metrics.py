import streamlit as st
import pandas as pd
import altair as alt
from core.storage import get_performance_df

st.title("Метрики времени инференса")
st.caption("Аппаратные замеры скорости генерации в потоковом режиме (streaming) на 10 задачах")

perf_df = get_performance_df()

# 1. Информационные карточки с описанием метрик
with st.container(border=True):
    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    with col_t1:
        st.markdown("**TTFT** *(Time to First Token)*")
        st.caption("Фаза prefill: время от отправки запроса до получения 1-го токена.")
    with col_t2:
        st.markdown("**TBT** *(Time Between Tokens)*")
        st.caption("Фаза decode: среднее время между токенами (~20 мс на GPU).")
    with col_t3:
        st.markdown("**Total Latency** *(U2E)*")
        st.caption("Полное время выполнения запроса от старта до финиша.")
    with col_t4:
        st.markdown("**Throughput**")
        st.caption("Пропускная способность: скорость генерации в токенах/сек.")

st.divider()

# 2. Таблица измерений с удобным переключателем вида
st.subheader("Таблица измерений по 10 промптам", anchor=False)

col_ctrl1, col_ctrl2 = st.columns([1, 2])
with col_ctrl1:
    view_type = st.segmented_control(
        "Режим таблицы:",
        options=["Основные метрики (TBT, TTFT, Throughput)", "Все параметры"],
        default="Основные метрики (TBT, TTFT, Throughput)"
    )

with col_ctrl2:
    category_filter = st.pills(
        "Фильтр по категориям:",
        options=["Все категории"] + sorted(perf_df["Категория"].unique().tolist()),
        default="Все категории"
    )

filtered_df = perf_df.copy()
if category_filter != "Все категории":
    filtered_df = filtered_df[filtered_df["Категория"] == category_filter]

if view_type == "Основные метрики (TBT, TTFT, Throughput)":
    compact_df = filtered_df[[
        "ID", "Категория",
        "TBT A (мс)", "TBT B (мс)",
        "TTFT A (с)", "TTFT B (с)",
        "Throughput A (т/с)", "Throughput B (т/с)"
    ]].copy()
    
    st.dataframe(
        compact_df,
        width="stretch",
        hide_index=True,
        column_config={
            "ID": st.column_config.NumberColumn("№", width="small"),
            "Категория": st.column_config.TextColumn("Категория задачи", width="medium"),
            "TBT A (мс)": st.column_config.NumberColumn("TBT Llama (мс)", format="%.2f"),
            "TBT B (мс)": st.column_config.NumberColumn("TBT Qwen (мс)", format="%.2f"),
            "TTFT A (с)": st.column_config.NumberColumn("TTFT Llama (с)", format="%.3f"),
            "TTFT B (с)": st.column_config.NumberColumn("TTFT Qwen (с)", format="%.3f"),
            "Throughput A (т/с)": st.column_config.NumberColumn("Throughput Llama (т/с)", format="%.1f"),
            "Throughput B (т/с)": st.column_config.NumberColumn("Throughput Qwen (т/с)", format="%.1f"),
        }
    )
else:
    st.dataframe(
        filtered_df,
        width="stretch",
        hide_index=True,
        column_config={
            "ID": st.column_config.NumberColumn("№", width="small"),
            "Категория": st.column_config.TextColumn("Категория задачи", width="medium"),
            "TBT A (мс)": st.column_config.NumberColumn("TBT Llama (мс)", format="%.2f"),
            "TBT B (мс)": st.column_config.NumberColumn("TBT Qwen (мс)", format="%.2f"),
            "TTFT A (с)": st.column_config.NumberColumn("TTFT Llama (с)", format="%.4f"),
            "TTFT B (с)": st.column_config.NumberColumn("TTFT Qwen (с)", format="%.4f"),
            "Latency A (с)": st.column_config.NumberColumn("Latency Llama (с)", format="%.3f"),
            "Latency B (с)": st.column_config.NumberColumn("Latency Qwen (с)", format="%.3f"),
            "Токены A": st.column_config.NumberColumn("Токены Llama"),
            "Токены B": st.column_config.NumberColumn("Токены Qwen"),
            "Throughput A (т/с)": st.column_config.NumberColumn("Throughput Llama (т/с)", format="%.1f"),
            "Throughput B (т/с)": st.column_config.NumberColumn("Throughput Qwen (т/с)", format="%.1f"),
        }
    )

st.divider()

# 3. Детальные вкладки анализа по каждой метрике
st.subheader("Сравнительный графический анализ", anchor=False)

tab_tbt, tab_ttft, tab_throughput = st.tabs([
    "TBT (Time Between Tokens)",
    "TTFT (Time to First Token)",
    "Throughput (Пропускная способность)"
])

# Вкладка TBT
with tab_tbt:
    st.markdown("#### Время декодирования одного токена (TBT / TPOT)")
    st.caption("Показывает скорость генерации каждого последующего токена после получения первого.")
    
    # 1. Таблица TBT для всех 10 промптов
    tbt_table_data = []
    for _, row in perf_df.iterrows():
        tbt_a = row["TBT A (мс)"]
        tbt_b = row["TBT B (мс)"]
        diff = round(tbt_b - tbt_a, 2)
        tok_speed_a = round(1000 / tbt_a, 1) if tbt_a > 0 else 0
        tok_speed_b = round(1000 / tbt_b, 1) if tbt_b > 0 else 0
        
        tbt_table_data.append({
            "№": row["ID"],
            "Категория": row["Категория"],
            "TBT Llama 3.2 (мс)": tbt_a,
            "TBT Qwen 2.5 (мс)": tbt_b,
            "Разница (мс)": diff,
            "Скорость Llama (ток/с)": tok_speed_a,
            "Скорость Qwen (ток/с)": tok_speed_b,
        })
    
    st.dataframe(
        pd.DataFrame(tbt_table_data),
        width="stretch",
        hide_index=True
    )
    
    st.write("")
    
    # 2. График TBT с корректным масштабом от 0
    df_tbt_chart = pd.DataFrame({
        "ID": list(perf_df["ID"]) * 2,
        "Категория": list(perf_df["Категория"]) * 2,
        "Модель": ["Llama 3.2 3B"] * len(perf_df) + ["Qwen 2.5 3B"] * len(perf_df),
        "TBT (мс)": list(perf_df["TBT A (мс)"]) + list(perf_df["TBT B (мс)"])
    })
    
    chart_tbt = alt.Chart(df_tbt_chart).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X("ID:O", title="Номер промпта", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("TBT (мс):Q", title="TBT (миллисекунды)", scale=alt.Scale(domain=[0, 25])),
        color=alt.Color("Модель:N", scale=alt.Scale(
            domain=["Llama 3.2 3B", "Qwen 2.5 3B"],
            range=["#18181B", "#71717A"]
        )),
        xOffset="Модель:N",
        tooltip=["ID", "Категория", "Модель", alt.Tooltip("TBT (мс):Q", format=".2f")]
    ).properties(height=300)
    
    st.altair_chart(chart_tbt, width="stretch")
    
    with st.container(border=True):
        st.markdown(
            "**Вывод по TBT:** Обе модели показывают идеальный паритет: **20.35 мс** (Llama) против **20.02 мс** (Qwen). "
            "Это означает, что декодер GPU NVIDIA GTX 1650 стабильно генерирует **~49–50 токенов в секунду** "
            "для обеих 3B-архитектур независимо от сложности задачи."
        )

# Вкладка TTFT
with tab_ttft:
    st.markdown("#### Время до первого токена (TTFT / Prefill Latency)")
    st.caption("Показывает задержку на чтение и обработку входного промпта моделью.")
    
    df_ttft_chart = pd.DataFrame({
        "ID": list(perf_df["ID"]) * 2,
        "Категория": list(perf_df["Категория"]) * 2,
        "Модель": ["Llama 3.2 3B"] * len(perf_df) + ["Qwen 2.5 3B"] * len(perf_df),
        "TTFT (с)": list(perf_df["TTFT A (с)"]) + list(perf_df["TTFT B (с)"])
    })
    
    chart_ttft = alt.Chart(df_ttft_chart).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X("ID:O", title="Номер промпта", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("TTFT (с):Q", title="TTFT (секунды)", scale=alt.Scale(domain=[0, 0.6])),
        color=alt.Color("Модель:N", scale=alt.Scale(
            domain=["Llama 3.2 3B", "Qwen 2.5 3B"],
            range=["#18181B", "#71717A"]
        )),
        xOffset="Модель:N",
        tooltip=["ID", "Категория", "Модель", alt.Tooltip("TTFT (с):Q", format=".4f")]
    ).properties(height=300)
    
    st.altair_chart(chart_ttft, width="stretch")
    
    with st.container(border=True):
        st.markdown(
            "**Вывод по TTFT:** На коротких промптах TTFT составляет всего **~0.19–0.22 с**. "
            "На промптах с длинным текстом (№2 — суммаризация статьи, №5 — извлечение сущностей) "
            "задержка возрастает до **0.45–0.51 с**. Это демонстрирует рост времени квадратичного внимания (prefill) от длины контекста."
        )

# Вкладка Throughput
with tab_throughput:
    st.markdown("#### Эффективная пропускная способность (Throughput)")
    st.caption("Количество сгенерированных токенов в единицу времени с учетом задержки TTFT.")
    
    df_thr_chart = pd.DataFrame({
        "ID": list(perf_df["ID"]) * 2,
        "Категория": list(perf_df["Категория"]) * 2,
        "Модель": ["Llama 3.2 3B"] * len(perf_df) + ["Qwen 2.5 3B"] * len(perf_df),
        "Throughput (т/с)": list(perf_df["Throughput A (т/с)"]) + list(perf_df["Throughput B (т/с)"])
    })
    
    chart_thr = alt.Chart(df_thr_chart).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X("ID:O", title="Номер промпта", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Throughput (т/с):Q", title="Пропускная способность (токенов/с)", scale=alt.Scale(domain=[0, 55])),
        color=alt.Color("Модель:N", scale=alt.Scale(
            domain=["Llama 3.2 3B", "Qwen 2.5 3B"],
            range=["#18181B", "#71717A"]
        )),
        xOffset="Модель:N",
        tooltip=["ID", "Категория", "Модель", alt.Tooltip("Throughput (т/с):Q", format=".1f")]
    ).properties(height=300)
    
    st.altair_chart(chart_thr, width="stretch")
    
    with st.container(border=True):
        st.markdown(
            "**Вывод по Throughput:** Средняя пропускная способность Qwen 2.5 3B составила **41.1 т/с**, "
            "а Llama 3.2 3B — **36.7 т/с**. На длинных ответах обе модели выходят на аппаратный лимит GPU в **47.5 т/с**."
        )
