import streamlit as st

st.set_page_config(
    page_title="LLM Benchmark • Лабораторная работа №2",
    page_icon=":material/analytics:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Определение страниц мультистраничного приложения
pages = [
    st.Page(
        "app_pages/overview.py",
        title="Обзор и стенд",
        icon=":material/dashboard:",
        default=True
    ),
    st.Page(
        "app_pages/speed_metrics.py",
        title="Метрики скорости",
        icon=":material/speed:"
    ),
    st.Page(
        "app_pages/quality_evaluation.py",
        title="Оценка качества",
        icon=":material/balance:"
    ),
    st.Page(
        "app_pages/live_benchmark.py",
        title="Интерактивный тест",
        icon=":material/play_arrow:"
    ),
    st.Page(
        "app_pages/conclusions.py",
        title="Итоговые выводы",
        icon=":material/article:"
    )
]

nav = st.navigation(pages)

with st.sidebar:
    st.title("Лабораторная №2")
    st.caption("Время инференса и качество языковых моделей")
    st.divider()
    
    with st.container(border=True):
        st.markdown("**Аппаратная платформа:**")
        st.caption("• GPU: NVIDIA GeForce GTX 1650 (4GB)")
        st.caption("• Backend: Ollama (100% VRAM offload)")
        st.caption("• Протокол: OpenAI REST API")
        
    with st.container(border=True):
        st.markdown("**Исследуемые модели:**")
        st.caption("1. `llama3.2:3b` (Meta)")
        st.caption("2. `qwen2.5:3b` (Alibaba Cloud)")
        st.caption("3. Судья: `qwen2.5:3b`")

nav.run()
