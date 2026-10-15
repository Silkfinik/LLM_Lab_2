import json
import pandas as pd

with open("benchmark_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

res_a = data["results_a"]
res_b = data["results_b"]
judge = data["judge_evaluations"]

# 1. Performance Table
perf_rows = []
for a, b in zip(res_a, res_b):
    perf_rows.append({
        "ID": a["id"],
        "Категория": a["category"],
        "TTFT A (с)": a["ttft_sec"],
        "TTFT B (с)": b["ttft_sec"],
        "TBT A (мс)": a["tbt_ms"],
        "TBT B (мс)": b["tbt_ms"],
        "Total Lat A (с)": a["total_latency_sec"],
        "Total Lat B (с)": b["total_latency_sec"],
        "Throughput A (т/с)": a["throughput_tok_s"],
        "Throughput B (т/с)": b["throughput_tok_s"],
        "Токенов A": a["tokens"],
        "Токенов B": b["tokens"],
    })

df_perf = pd.DataFrame(perf_rows)
print("--- СКОРОСТЬ И ИНФЕРЕНС ---")
print(df_perf.to_string(index=False))

# Averages
print("\n--- СРЕДНИЕ ЗНАЧЕНИЯ СКОРОСТИ ---")
print(f"Llama 3.2 3B (Модель A):")
print(f"  Средний TTFT: {df_perf['TTFT A (с)'].mean():.4f} с")
print(f"  Средний TBT: {df_perf['TBT A (мс)'].mean():.2f} мс")
print(f"  Средний Latency: {df_perf['Total Lat A (с)'].mean():.4f} с")
print(f"  Средний Throughput: {df_perf['Throughput A (т/с)'].mean():.2f} токенов/сек")

print(f"\nQwen 2.5 3B (Модель B):")
print(f"  Средний TTFT: {df_perf['TTFT B (с)'].mean():.4f} с")
print(f"  Средний TBT: {df_perf['TBT B (мс)'].mean():.2f} мс")
print(f"  Средний Latency: {df_perf['Total Lat B (с)'].mean():.4f} с")
print(f"  Средний Throughput: {df_perf['Throughput B (т/с)'].mean():.2f} токенов/сек")

# 2. Quality Table (BLEU & ROUGE)
qual_rows = []
for a, b in zip(res_a, res_b):
    qual_rows.append({
        "ID": a["id"],
        "Категория": a["category"],
        "BLEU A": a["scores"]["bleu"],
        "BLEU B": b["scores"]["bleu"],
        "ROUGE-1 A": a["scores"]["rouge1"],
        "ROUGE-1 B": b["scores"]["rouge1"],
        "ROUGE-L A": a["scores"]["rougeL"],
        "ROUGE-L B": b["scores"]["rougeL"],
    })

df_qual = pd.DataFrame(qual_rows)
print("\n--- КАЧЕСТВО (BLEU & ROUGE) ---")
print(df_qual.to_string(index=False))

print("\n--- СРЕДНИЕ ЗНАЧЕНИЯ КАЧЕСТВА ---")
print(f"Llama 3.2 3B (Модель A): BLEU = {df_qual['BLEU A'].mean():.2f}, ROUGE-1 = {df_qual['ROUGE-1 A'].mean():.2f}, ROUGE-L = {df_qual['ROUGE-L A'].mean():.2f}")
print(f"Qwen 2.5 3B (Модель B): BLEU = {df_qual['BLEU B'].mean():.2f}, ROUGE-1 = {df_qual['ROUGE-1 B'].mean():.2f}, ROUGE-L = {df_qual['ROUGE-L B'].mean():.2f}")

# 3. Judge Verdicts
print("\n--- СУДЕЙСТВО LLM-AS-A-JUDGE ---")
for j in judge:
    print(f"\n[Промпт {j['id']}: {j['category']}]")
    print(j['verdict'])
