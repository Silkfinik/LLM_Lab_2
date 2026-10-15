"""
Модуль загрузки и структурирования данных бенчмарка.
"""
import os
import json
import re
from typing import Dict, Any, List, Tuple
import pandas as pd

RESULTS_PATH = "data/benchmark_results.json"

def load_raw_data() -> Dict[str, Any]:
    """Загружает сырые данные эксперимента из JSON."""
    if not os.path.exists(RESULTS_PATH):
        # Fallback на корень если запускается из другого каталога
        if os.path.exists("benchmark_results.json"):
            with open("benchmark_results.json", "r", encoding="utf-8") as f:
                return json.load(f)
        raise FileNotFoundError(f"Файл результатов {RESULTS_PATH} не найден.")

    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def get_performance_df() -> pd.DataFrame:
    """Формирует DataFrame со всеми метриками скорости инференса."""
    data = load_raw_data()
    res_a = data["results_a"]
    res_b = data["results_b"]

    rows = []
    for a, b in zip(res_a, res_b):
        rows.append({
            "ID": a["id"],
            "Категория": a["category"],
            "TTFT A (с)": a["ttft_sec"],
            "TTFT B (с)": b["ttft_sec"],
            "TBT A (мс)": a["tbt_ms"],
            "TBT B (мс)": b["tbt_ms"],
            "Latency A (с)": a["total_latency_sec"],
            "Latency B (с)": b["total_latency_sec"],
            "Токены A": a["tokens"],
            "Токены B": b["tokens"],
            "Throughput A (т/с)": a["throughput_tok_s"],
            "Throughput B (т/с)": b["throughput_tok_s"]
        })
    return pd.DataFrame(rows)

def get_quality_df() -> pd.DataFrame:
    """Формирует DataFrame с автоматическими метриками качества BLEU и ROUGE."""
    data = load_raw_data()
    res_a = data["results_a"]
    res_b = data["results_b"]

    rows = []
    for a, b in zip(res_a, res_b):
        rows.append({
            "ID": a["id"],
            "Категория": a["category"],
            "BLEU A": a["scores"]["bleu"],
            "BLEU B": b["scores"]["bleu"],
            "ROUGE-1 A": a["scores"]["rouge1"],
            "ROUGE-1 B": b["scores"]["rouge1"],
            "ROUGE-L A": a["scores"]["rougeL"],
            "ROUGE-L B": b["scores"]["rougeL"]
        })
    return pd.DataFrame(rows)

def get_judge_records() -> List[Dict[str, Any]]:
    """Возвращает детальные записи слепого судейства LLM-as-a-Judge."""
    data = load_raw_data()
    res_a = data["results_a"]
    res_b = data["results_b"]
    judge = data["judge_evaluations"]

    records = []
    for a, b, j in zip(res_a, res_b, judge):
        v = j["verdict"]
        score_a_match = re.search(r"Оценка А:\s*(\d+)", v)
        score_b_match = re.search(r"Оценка Б:\s*(\d+)", v)
        winner_match = re.search(r"Победитель:\s*(Ответ [АБ]|Ничья)", v)

        score_a = int(score_a_match.group(1)) if score_a_match else 0
        score_b = int(score_b_match.group(1)) if score_b_match else 0
        winner = winner_match.group(1) if winner_match else "Не определен"

        records.append({
            "id": a["id"],
            "category": a["category"],
            "prompt": a["prompt"],
            "reference": a["reference"],
            "answer_a": a["text"],
            "answer_b": b["text"],
            "score_a": score_a,
            "score_b": score_b,
            "winner": winner,
            "verdict_raw": v
        })
    return records

def get_kpi_summary() -> Dict[str, Any]:
    """Сводные агрегированные KPI для главного экрана."""
    perf_df = get_performance_df()
    qual_df = get_quality_df()
    judge_records = get_judge_records()

    wins_a = sum(1 for r in judge_records if "Ответ А" in r["winner"])
    wins_b = sum(1 for r in judge_records if "Ответ Б" in r["winner"])
    ties = sum(1 for r in judge_records if "Ничья" in r["winner"])

    return {
        "mean_ttft_a": round(perf_df["TTFT A (с)"].mean(), 3),
        "mean_ttft_b": round(perf_df["TTFT B (с)"].mean(), 3),
        "mean_tbt_a": round(perf_df["TBT A (мс)"].mean(), 2),
        "mean_tbt_b": round(perf_df["TBT B (мс)"].mean(), 2),
        "mean_throughput_a": round(perf_df["Throughput A (т/с)"].mean(), 1),
        "mean_throughput_b": round(perf_df["Throughput B (т/с)"].mean(), 1),
        "mean_bleu_a": round(qual_df["BLEU A"].mean(), 2),
        "mean_bleu_b": round(qual_df["BLEU B"].mean(), 2),
        "mean_rouge_l_a": round(qual_df["ROUGE-L A"].mean(), 2),
        "mean_rouge_l_b": round(qual_df["ROUGE-L B"].mean(), 2),
        "judge_mean_a": round(sum(r["score_a"] for r in judge_records) / len(judge_records), 1),
        "judge_mean_b": round(sum(r["score_b"] for r in judge_records) / len(judge_records), 1),
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties
    }
