import time
import json
import random
from typing import List, Dict, Any
from openai import OpenAI
import evaluate
import sacrebleu
import pandas as pd

# 1. Configuration
BASE_URL = "http://localhost:11434/v1"
API_KEY = "ollama"
MODEL_A = "llama3.2:3b"
MODEL_B = "qwen2.5:3b"
JUDGE_MODEL = "qwen2.5:3b"

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# 2. Prompts and Reference Answers (10 diverse prompts)
PROMPTS_DATA = [
    {
        "id": 1,
        "category": "Фактический вопрос (QA)",
        "prompt": "В каком году человек впервые полетел в космос, и как звали этого космонавта? Ответь одним кратким предложением.",
        "reference": "Человек впервые полетел в космос 12 апреля 1961 года, этим космонавтом был Юрий Гагарин."
    },
    {
        "id": 2,
        "category": "Суммаризация (Summarization)",
        "prompt": "Сократи следующий текст до одного информативного предложения: 'Искусственный интеллект активно внедряется в медицину. Нейросети помогают врачам анализировать рентгеновские снимки, томограммы и выявлять патологии на ранних стадиях с высокой точностью. Это существенно снижает нагрузку на медицинский персонал и ускоряет постановку диагнозов.'",
        "reference": "Искусственный интеллект помогает врачам точнее и быстрее диагностировать заболевания по медицинским снимкам, снижая нагрузку на персонал."
    },
    {
        "id": 3,
        "category": "Перевод (Translation)",
        "prompt": "Переведи на русский язык следующее предложение: 'Large language models are transforming natural language processing by enabling zero-shot and few-shot task generalization.'",
        "reference": "Большие языковые модели трансформируют обработку естественного языка, обеспечивая обобщение задач в режимах zero-shot и few-shot."
    },
    {
        "id": 4,
        "category": "Логическая задача (Logic)",
        "prompt": "У фермера было 17 овец. Все, кроме 9, убежали. Сколько овец осталось у фермера? Объясни свой ответ в одно предложение.",
        "reference": "У фермера осталось 9 овец, так как по условию фраза 'все, кроме 9' означает, что именно девять овец не убежали."
    },
    {
        "id": 5,
        "category": "Извлечение информации (Extraction)",
        "prompt": "Прочитай текст и назови компанию-разработчика, название процессора и год его выпуска: 'В 1971 году компания Intel представила первый в мире коммерческий однокристальный микропроцессор Intel 4004, ознаменовав начало эры микрокомпьютеров.' Ответь строго в формате: 'Компания: ..., Процессор: ..., Год: ...'.",
        "reference": "Компания: Intel, Процессор: Intel 4004, Год: 1971."
    },
    {
        "id": 6,
        "category": "Программирование (Python code)",
        "prompt": "Напиши функцию на Python с именем `is_palindrome(s: str) -> bool`, которая проверяет, является ли строка палиндромом, игнорируя регистр и пробелы. Приведи только код функции.",
        "reference": "def is_palindrome(s: str) -> bool:\n    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n    return cleaned == cleaned[::-1]"
    },
    {
        "id": 7,
        "category": "Анализ / Концепция (Concept QA)",
        "prompt": "В чем главное техническое различие между потоком (thread) и процессом (process) в операционной системе? Назови кратко 2 ключевых отличия.",
        "reference": "Главные отличия: процессы изолированы и имеют собственное адресное пространство памяти, а потоки одного процесса разделяют общую память; создание и переключение контекста потоков происходит быстрее и требует меньше ресурсов."
    },
    {
        "id": 8,
        "category": "Структурирование данных (JSON)",
        "prompt": "Преобразуй следующие данные в валидный JSON: 'Иван Иванов, 25 лет, инженер, навыки: Python, SQL'. Верни только JSON без лишних пояснений.",
        "reference": '{\n  "name": "Иван Иванов",\n  "age": 25,\n  "profession": "инженер",\n  "skills": ["Python", "SQL"]\n}'
    },
    {
        "id": 9,
        "category": "Креативное объяснение (Creative)",
        "prompt": "Придумай метафору, объясняющую принцип работы механизма внимания (Attention) в нейросетях-трансформерах человеку без технического образования (не более 3 предложений).",
        "reference": "Механизм внимания похож на луч фонарика в темной комнате или на читателя с маркером. Читая каждое новое слово, модель подсвечивает наиболее важные связанные слова в контексте, вместо того чтобы одинаково помнить весь текст."
    },
    {
        "id": 10,
        "category": "Пошаговый алгоритм (Step-by-step)",
        "prompt": "Опиши по шагам классический алгоритм бинарного поиска в отсортированном массиве (кратко, не более 4 шагов).",
        "reference": "1. Определить границы поиска (начало и конец массива) и найти индекс среднего элемента. 2. Сравнить средний элемент с искомым значением; если они равны — элемент найден. 3. Если искомое меньше среднего — сдвинуть правую границу влево от середины, если больше — сдвинуть левую границу вправо. 4. Повторять шаги 1–3, пока границы не сойдутся или элемент не будет найден."
    }
]

def warm_up(model_name: str):
    print(f"\n[Warm-up] Прогрев модели '{model_name}' в VRAM...")
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Привет, подтверди готовность одним словом."}],
            max_tokens=10
        )
        print(f"[Warm-up] Модель '{model_name}' готова к инференсу.")
    except Exception as e:
        print(f"[Warm-up Error] {e}")

def measure_inference(model_name: str, prompt: str) -> Dict[str, Any]:
    """
    Замеряет метрики инференса в streaming режиме:
    - TTFT (Time to First Token)
    - Total Latency (U2E)
    - TBT / TPOT (Time Between Tokens)
    - Throughput (tokens/s)
    """
    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        stream_options={"include_usage": True}
    )

    first_token_time = None
    chunks = []
    usage = None

    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            chunks.append(chunk.choices[0].delta.content)
        if hasattr(chunk, "usage") and chunk.usage:
            usage = chunk.usage

    t_end = time.perf_counter()
    full_text = "".join(chunks).strip()
    total_latency = t_end - t0

    if first_token_time is not None:
        ttft = first_token_time - t0
    else:
        ttft = total_latency

    if usage and usage.completion_tokens:
        tokens_generated = usage.completion_tokens
    else:
        # Резервный подсчет приблизительных токенов если usage пуст
        tokens_generated = max(1, len(full_text.split()) * 4 // 3)

    # TBT / TPOT: среднее время между токенами на этапе decode
    if tokens_generated > 1:
        tbt = (total_latency - ttft) / (tokens_generated - 1)
    else:
        tbt = 0.0

    throughput = tokens_generated / total_latency if total_latency > 0 else 0.0

    return {
        "text": full_text,
        "ttft_sec": round(ttft, 4),
        "total_latency_sec": round(total_latency, 4),
        "tbt_ms": round(tbt * 1000, 2),  # в миллисекундах
        "throughput_tok_s": round(throughput, 2),
        "tokens": tokens_generated
    }

def run_benchmarks():
    print("=" * 60)
    print("ЭТАП 1: Замер метрик скорости инференса для обеих моделей")
    print("=" * 60)

    # Запускаем сначала Модель А для всех 10 промптов (чтобы не перегружать VRAM туда-обратно)
    warm_up(MODEL_A)
    results_a = []
    print(f"\nЗапуск тестов для {MODEL_A}...")
    for item in PROMPTS_DATA:
        print(f"  [{MODEL_A}] Промпт {item['id']}/10: {item['category']}...")
        metrics = measure_inference(MODEL_A, item["prompt"])
        results_a.append({
            "id": item["id"],
            "category": item["category"],
            "prompt": item["prompt"],
            "reference": item["reference"],
            **metrics
        })

    # Затем Модель Б для всех 10 промптов
    warm_up(MODEL_B)
    results_b = []
    print(f"\nЗапуск тестов для {MODEL_B}...")
    for item in PROMPTS_DATA:
        print(f"  [{MODEL_B}] Промпт {item['id']}/10: {item['category']}...")
        metrics = measure_inference(MODEL_B, item["prompt"])
        results_b.append({
            "id": item["id"],
            "category": item["category"],
            "prompt": item["prompt"],
            "reference": item["reference"],
            **metrics
        })

    print("\n" + "=" * 60)
    print("ЭТАП 2: Оценка качества методом BLEU и ROUGE")
    print("=" * 60)

    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")

    def calc_scores(prediction: str, reference: str):
        # sacrebleu sentence bleu с экспоненциальным сглаживанием
        b_score = sacrebleu.sentence_bleu(prediction, [reference], smooth_method="exp").score
        # rouge с токенизатором по словам для поддержки русского языка
        r_scores = rouge_metric.compute(
            predictions=[prediction],
            references=[reference],
            tokenizer=lambda x: x.lower().split()
        )
        return {
            "bleu": round(b_score, 2),
            "rouge1": round(r_scores["rouge1"] * 100, 2),
            "rouge2": round(r_scores["rouge2"] * 100, 2),
            "rougeL": round(r_scores["rougeL"] * 100, 2)
        }

    for res_a, res_b in zip(results_a, results_b):
        scores_a = calc_scores(res_a["text"], res_a["reference"])
        scores_b = calc_scores(res_b["text"], res_b["reference"])
        res_a["scores"] = scores_a
        res_b["scores"] = scores_b

    print("BLEU и ROUGE успешно рассчитаны для всех промптов.")

    print("\n" + "=" * 60)
    print("ЭТАП 3: Оценка методом LLM-as-a-Judge (Слепое сравнение)")
    print("=" * 60)
    warm_up(JUDGE_MODEL)

    judge_evaluations = []
    for item_a, item_b in zip(results_a, results_b):
        p_id = item_a["id"]
        prompt = item_a["prompt"]
        ans_a = item_a["text"]
        ans_b = item_b["text"]

        judge_prompt = f"""Ты — беспристрастный эксперт и судья качества ответов языковых моделей.
Оцени два анонимных ответа на следующий запрос пользователя:

ЗАПРОС ПОЛЬЗОВАТЕЛЯ:
{prompt}

ЭТАЛОННЫЙ ОТВЕТ (для ориентира):
{item_a["reference"]}

ОТВЕТ А:
{ans_a}

ОТВЕТ Б:
{ans_b}

ИНСТРУКЦИЯ:
1. Оцени Ответ А по шкале от 1 до 10 (критерии: точность, соответствие промпту, связность).
2. Оцени Ответ Б по шкале от 1 до 10.
3. Выбери победителя: строго напиши 'Ответ А', 'Ответ Б' или 'Ничья'.
4. Дай краткое обоснование (2-3 предложения).

ФОРМАТ ОТВЕТА (строго соблюдай):
Оценка А: <число от 1 до 10>
Оценка Б: <число от 1 до 10>
Победитель: <Ответ А / Ответ Б / Ничья>
Обоснование: <текст>
"""
        print(f"  Судейство для промпта {p_id}/10...")
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": "Ты независимый профессиональный судья для оценки качества текста LLM."},
                    {"role": "user", "content": judge_prompt}
                ],
                temperature=0.2,
                max_tokens=400
            )
            verdict = resp.choices[0].message.content.strip()
        except Exception as e:
            verdict = f"Ошибка судейства: {e}"

        judge_evaluations.append({
            "id": p_id,
            "category": item_a["category"],
            "prompt": prompt,
            "verdict": verdict
        })

    # Сохраняем все сырые данные в JSON
    output_data = {
        "model_a": MODEL_A,
        "model_b": MODEL_B,
        "judge_model": JUDGE_MODEL,
        "results_a": results_a,
        "results_b": results_b,
        "judge_evaluations": judge_evaluations
    }
    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print("\n[Готово] Данные сохранены в benchmark_results.json")
    return output_data

if __name__ == "__main__":
    run_benchmarks()
