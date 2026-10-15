"""Displays actual saved data and coverage; never calls models."""
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


def render():
    st.set_page_config(page_title="Локальные LLM · Эксперименты", page_icon="📊", layout="wide")
    st.title("Скорость и качество локальных LLM")
    st.caption("Повторные измерения · Проверки заданий · Прозрачные условия эксперимента")
    runs = Path(__file__).resolve().parent.parent / "runs"
    summaries = sorted(runs.glob("*/analysis/summary.json")) if runs.exists() else []
    if not summaries:
        st.info("Новые результаты ещё не получены. Выполните команды из RUN_EXPERIMENTS.md, затем research.py analyze.")
        st.markdown("Старые одиночные измерения сохранены в benchmark_results.json и не подмешиваются в новые серии.")
        return
    selected = st.sidebar.selectbox("Серия экспериментов", summaries, format_func=lambda p: p.parent.parent.name)
    if st.sidebar.button("Обновить данные"):
        st.rerun()
    summary = json.loads(selected.read_text(encoding="utf-8"))
    manifest = json.loads((selected.parent.parent / "manifest.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(selected.parent / "responses.csv")
    models = frame["model"].drop_duplicates().tolist()
    colors = ["#2563eb", "#e87924"]
    a, b, c = st.columns(3)
    a.metric("Получено запросов", f"{summary['received']} / {summary['planned']}")
    b.metric("Серия", summary["suite"])
    c.metric("Функциональные тесты кода", "Включены" if summary["execute_code"] else "Не выполнялись")
    if summary["received"] != summary["planned"]:
        st.warning("В серии есть ошибки или пропуски. Проверьте coverage перед сравнением средних.")
    temperature = st.sidebar.selectbox("Температура", sorted(frame["temperature"].dropna().unique()))
    filtered = frame[frame["temperature"] == temperature].copy()
    tab_overview, tab_quality, tab_speed, tab_answers = st.tabs(["Сводка", "Качество", "Скорость", "Ответы и экспорт"])
    with tab_overview:
        st.subheader("Условия запуска")
        st.json(manifest["identity"]["settings"], expanded=False)
        st.json(manifest["environment"], expanded=False)
        st.caption("Интервалы качества рассчитаны по задачам, а не по отдельным повторным генерациям.")
        table = pd.read_csv(selected.parent / "summary.csv")
        st.dataframe(table[table["temperature"] == temperature], hide_index=True, use_container_width=True)
        st.subheader("Парная разница качества")
        st.dataframe(pd.DataFrame(summary["paired_quality"]), hide_index=True, use_container_width=True)
    with tab_quality:
        image = selected.parent / "figures" / f"quality_t{temperature:g}.png"
        if image.exists():
            st.image(str(image), use_container_width=True)
        if "rougeL_f1" in filtered:
            chart = alt.Chart(filtered.dropna(subset=["rougeL_f1"])).mark_circle(size=75, opacity=.65).encode(
                x=alt.X("category:N", title="Категория", axis=alt.Axis(labelAngle=-30)),
                y=alt.Y("rougeL_f1:Q", title="ROUGE-L F1, %", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color("model:N", title="Модель", scale=alt.Scale(domain=models, range=colors)),
                tooltip=["model", "task_id", "repeat", "rougeL_f1", "evaluation_status"])
            st.altair_chart(chart.properties(height=340), use_container_width=True)
        st.info("BLEU/ROUGE измеряют лексическое сходство. Открытые задания требуют независимой оценки; валидный JSON не гарантирует правильное содержание.")
        if "evaluation_status" in filtered:
            st.dataframe(filtered.groupby(["model", "evaluation_status"], dropna=False).size().reset_index(name="Количество"), hide_index=True)
    with tab_speed:
        st.caption("TTFT — задержка до первого текстового фрагмента. Серверное время на токен — eval_duration / eval_count. Границы фрагментов не считаются границами токенов.")
        good = filtered[filtered["status"] == "ok"].copy()
        if not good.empty:
            good = good[good["context"] == st.selectbox("Рабочий контекст", sorted(good["context"].unique()))]
            good = good[good["budget"] == st.selectbox("Лимит выходных токенов", sorted(good["budget"].unique()))]
            for metric, label in [("ttft_sec", "TTFT, с"), ("server_tpot_ms", "Серверное время на токен, мс")]:
                if metric in good:
                    chart = alt.Chart(good.dropna(subset=[metric])).mark_circle(size=85, opacity=.65).encode(
                        x=alt.X("prompt_tokens:Q", title="Фактические входные токены"),
                        y=alt.Y(f"{metric}:Q", title=label, scale=alt.Scale(zero=True)),
                        color=alt.Color("model:N", scale=alt.Scale(domain=models, range=colors)),
                        tooltip=["model", "task_id", "repeat", "tokens", "done_reason", metric])
                    st.altair_chart(chart.properties(height=280), use_container_width=True)
        try:
            conditions = pd.read_csv(selected.parent / "speed_by_condition.csv")
            st.dataframe(conditions[conditions["temperature"] == temperature], hide_index=True, use_container_width=True)
        except pd.errors.EmptyDataError:
            st.info("Успешных измерений скорости нет; проверьте raw.jsonl.")
    with tab_answers:
        statuses = sorted(filtered["status"].unique())
        chosen = st.multiselect("Состояние запроса", statuses, default=statuses)
        st.dataframe(filtered[filtered["status"].isin(chosen)], hide_index=True, use_container_width=True)
        for filename in ["summary.json", "summary.csv", "responses.csv", "REPORT.md"]:
            path = selected.parent / filename
            st.download_button(f"Скачать {filename}", path.read_bytes(), file_name=filename)
        st.json(summary["caveats"], expanded=False)
