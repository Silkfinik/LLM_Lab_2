import streamlit as st
import pandas as pd
import altair as alt
from core.storage import get_quality_df, get_judge_records, get_kpi_summary

st.title("Сравнение качества ответов")
st.caption("Анализ качества генерации двумя методами: автоматические метрики (BLEU, ROUGE) и слепое судейство (LLM-as-a-Judge)")

kpis = get_kpi_summary()
qual_df = get_quality_df()
judge_records = get_judge_records()

view_mode = st.segmented_control(
    "Раздел оценки",
    options=["Слепое судейство (LLM-as-a-Judge)", "Метрики BLEU и ROUGE", "Сопоставление подходов"],
    default="Слепое судейство (LLM-as-a-Judge)"
)

st.divider()

if view_mode == "Слепое судейство (LLM-as-a-Judge)":
    # 1. Сводка судейства
    col_w1, col_w2, col_w3 = st.columns(3)
    with col_w1:
        with st.container(border=True):
            st.metric("Побед Qwen 2.5 3B", f"{kpis['wins_b']} из 10", "80% всех раундов")
    with col_w2:
        with st.container(border=True):
            st.metric("Побед Llama 3.2 3B", f"{kpis['wins_a']} из 10", "10% всех раундов")
    with col_w3:
        with st.container(border=True):
            st.metric("Ничьих", f"{kpis['ties']} из 10", "10% всех раундов")

    st.subheader("Слепые карточки по каждому заданию", anchor=False)
    st.caption("Нажмите на карточку, чтобы посмотреть сгенерированные ответы и разбор судьи:")

    for r in judge_records:
        winner_label = r["winner"]
        if "Ответ Б" in winner_label:
            badge = "Победитель: Qwen 2.5 3B (Ответ Б)"
        elif "Ответ А" in winner_label:
            badge = "Победитель: Llama 3.2 3B (Ответ А)"
        else:
            badge = "Результат: Ничья"

        title = f"Промпт №{r['id']}: {r['category']} — [{badge}]"
        
        with st.expander(title):
            st.markdown(f"**Запрос пользователя:** `{r['prompt']}`")
            st.markdown(f"**Эталонный ответ:** *{r['reference']}*")
            st.write("")

            col_a, col_b = st.columns(2)
            with col_a:
                with st.container(border=True):
                    st.markdown("**Ответ А (Llama 3.2 3B)**")
                    st.text(r["answer_a"])
                    st.caption(f"Оценка арбитра: **{r['score_a']} / 10**")

            with col_b:
                with st.container(border=True):
                    st.markdown("**Ответ Б (Qwen 2.5 3B)**")
                    st.text(r["answer_b"])
                    st.caption(f"Оценка арбитра: **{r['score_b']} / 10**")

            st.markdown("##### Полный вердикт модели-судьи:")
            with st.container(border=True):
                st.markdown(r["verdict_raw"])

elif view_mode == "Метрики BLEU и ROUGE":
    st.subheader("Автоматические метрики попарного совпадения n-грамм", anchor=False)
    st.markdown(
        """
        * **BLEU (sacrebleu):** точность (precision) совпадения n-грамм ответа с эталоном со штрафом за краткость.
        * **ROUGE-1 / ROUGE-L:** полнота (recall) по униграммам и наибольшей общей подпоследовательности (LCS).
        * *Токенизация выполнена с учетом русской морфологии и регистра.*
        """
    )

    st.dataframe(
        qual_df,
        width="stretch",
        hide_index=True,
        column_config={
            "ID": st.column_config.NumberColumn("№", width="small"),
            "Категория": st.column_config.TextColumn("Категория", width="medium"),
            "BLEU A": st.column_config.NumberColumn("BLEU Llama", format="%.2f"),
            "BLEU B": st.column_config.NumberColumn("BLEU Qwen", format="%.2f"),
            "ROUGE-1 A": st.column_config.NumberColumn("ROUGE-1 Llama (%)", format="%.2f"),
            "ROUGE-1 B": st.column_config.NumberColumn("ROUGE-1 Qwen (%)", format="%.2f"),
            "ROUGE-L A": st.column_config.NumberColumn("ROUGE-L Llama (%)", format="%.2f"),
            "ROUGE-L B": st.column_config.NumberColumn("ROUGE-L Qwen (%)", format="%.2f"),
        }
    )

    st.divider()

    st.subheader("Сравнение ROUGE-L по задачам", anchor=False)
    df_chart = pd.DataFrame({
        "ID": list(qual_df["ID"]) * 2,
        "Категория": list(qual_df["Категория"]) * 2,
        "Модель": ["Llama 3.2 3B"] * len(qual_df) + ["Qwen 2.5 3B"] * len(qual_df),
        "ROUGE-L": list(qual_df["ROUGE-L A"]) + list(qual_df["ROUGE-L B"])
    })

    chart = alt.Chart(df_chart).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X("ID:O", title="Номер промпта", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("ROUGE-L:Q", title="ROUGE-L Score (%)"),
        color=alt.Color("Модель:N", scale=alt.Scale(
            domain=["Llama 3.2 3B", "Qwen 2.5 3B"],
            range=["#18181B", "#71717A"]
        )),
        xOffset="Модель:N",
        tooltip=["ID", "Категория", "Модель", alt.Tooltip("ROUGE-L:Q", format=".2f")]
    ).properties(height=320)

    st.altair_chart(chart, width="stretch")

else:
    st.subheader("Сопоставление автоматических метрик и LLM-as-a-Judge", anchor=False)
    
    with st.container(border=True):
        st.markdown(
            """
            ### Ключевые методологические выводы:

            1. **Ограничения n-граммных метрик (BLEU / ROUGE):**
               * На открытых творческих, логических и аналитических задачах (промпты №7 и №9) метрики BLEU и ROUGE показывают околонулевые значения (**0.3–0.6**), несмотря на то что ответы обеих моделей были содержательными и верными.
               * Метрики наказывают модель за синонимы, изменения порядка слов и другие вариации формулировок.
            
            2. **Преимущество LLM-as-a-Judge:**
               * Судья оценивает **смысловую корректность и логику**, а не поверхностное совпадение слов.
               * Например, в логической задаче №4 Llama написала «осталось 8 овец» (ошибка), а Qwen — «осталось 9 овец» (верно). Судья моментально зафиксировал смысловую ошибку (оценки 3 против 8), в то время как поверхностные метрики могли бы засчитать общие служебные слова.
            
            3. **Систематические расхождения:**
               * В задачах жесткого извлечения информации (промпт №5) BLEU и ROUGE показали максимальные баллы (до 100%), что совпало с высокими оценками арбитра (9 и 8 баллов).
               * Таким образом, **BLEU/ROUGE эффективны для RAG и структурированных фактов**, а **LLM-as-a-Judge незаменим для рассуждений, диалога и кода**.
            """
        )
