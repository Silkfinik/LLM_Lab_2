"""
Модуль расчета классических метрик качества текста (BLEU и ROUGE).
"""
from typing import Dict
import evaluate
import sacrebleu

_bleu_metric = None
_rouge_metric = None

def _get_metrics():
    global _bleu_metric, _rouge_metric
    if _bleu_metric is None:
        _bleu_metric = evaluate.load("bleu")
    if _rouge_metric is None:
        _rouge_metric = evaluate.load("rouge")
    return _bleu_metric, _rouge_metric

def compute_quality_scores(prediction: str, reference: str) -> Dict[str, float]:
    """
    Вычисляет BLEU (sacrebleu со сглаживанием) и ROUGE (1, 2, L)
    с разбиением по словам для корректной обработки русского текста.
    """
    _, rouge = _get_metrics()

    # sacrebleu sentence bleu с экспоненциальным сглаживанием
    b_score = sacrebleu.sentence_bleu(prediction, [reference], smooth_method="exp").score

    # rouge с токенизацией по пробелам/словам в нижнем регистре
    r_scores = rouge.compute(
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
