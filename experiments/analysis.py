"""Offline validation, task-clustered intervals, tables and publication figures."""
import csv
import hashlib
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from .checks import check_task, reference_metrics
from .runner import now, read_jsonl, write_json


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * q
    low = int(position)
    fraction = position - low
    return values[low] * (1 - fraction) + values[min(low + 1, len(values) - 1)] * fraction


def interval(values, seed=2026):
    if not values:
        return dict(mean=None, median=None, p95=None, ci_low=None, ci_high=None, n=0)
    rng = random.Random(seed)
    samples = [statistics.mean(rng.choices(values, k=len(values))) for _ in range(2000)]
    return dict(mean=statistics.mean(values), median=statistics.median(values),
                p95=percentile(values, .95), ci_low=percentile(samples, .025),
                ci_high=percentile(samples, .975), n=len(values))


def write_csv(path, rows):
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(args):
    directory = Path(args.run).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    plan = json.loads((directory / "plan.json").read_text())
    raw_path = directory / "raw.jsonl"
    latest = {r["job_id"]: r for r in read_jsonl(raw_path)}
    output = directory / "analysis"
    output.mkdir(exist_ok=True)
    if args.execute_code:
        from .runner import command_output
        for cmd in (["docker", "info"], ["docker", "image", "inspect", args.docker_image]):
            result = command_output(cmd)
            if result.get("returncode", 1) != 0:
                raise RuntimeError("Docker or the local code-test image is unavailable; see RUN_EXPERIMENTS.md")
    records = []
    for job in plan:
        row = latest.get(job["job_id"])
        base = {k: job[k] for k in ("job_id", "model", "repeat", "temperature", "context", "budget", "cold")}
        base.update(suite=manifest["suite"], task_id=job["task"]["id"], category=job["task"]["category"],
                    check_kind=job["task"].get("check"),
                    status=row["status"] if row else "missing")
        if not row or row["status"] != "ok":
            base.update(evaluation_status="request_" + base["status"],
                        task_pass=False if job["task"].get("check") else None,
                        error=row.get("error") if row else "not measured")
        else:
            response = row["response"]
            base.update({k: v for k, v in response.items()
                         if k not in ("text", "thinking", "stream_events", "raw_final", "request")})
            base.update(answer=response["text"], reference=job["task"]["reference"])
            if job["task"]["reference"]:
                base.update(reference_metrics(response["text"], job["task"]["reference"]))
                base.update(check_task(job["task"], response["text"], args.execute_code, args.docker_image))
                if response["truncated"] and base.get("task_pass"):
                    base["task_pass"] = False
                    base["evaluation_status"] = "truncated"
        records.append(base)
    write_json(output / "evaluated.json", records)
    write_csv(output / "responses.csv", [{k: v for k, v in r.items() if not isinstance(v, (list, dict))} for r in records])
    groups = defaultdict(list)
    for row in records:
        groups[(row["model"], row["temperature"], row["context"], row["budget"])].append(row)
    summaries = []
    for (model, temperature, context, budget), group in groups.items():
        success = [r for r in group if r["status"] == "ok"]
        summary = dict(model=model, temperature=temperature, context=context, budget=budget,
                       planned=len(group), received=len(success), errors=len(group) - len(success),
                       truncated=sum(r.get("truncated", False) for r in success))
        for metric in ["ttft_sec", "total_latency_sec", "server_tpot_ms", "client_tpot_ms",
                       "server_decode_tok_s", "throughput_tok_s", "chars_per_sec", "tokens", "load_sec"]:
            values = [r[metric] for r in success if r.get(metric) is not None]
            summary[metric] = interval(values)
        # Repeated generations of the same task are NOT independent tasks for quality CIs.
        for metric in ["sentence_bleu", "rouge1_f1", "rougeL_f1", "task_pass", "format_pass", "functional_pass"]:
            by_task = defaultdict(list)
            for row in group:
                if metric == "task_pass" and row.get("check_kind") == "code" and not args.execute_code:
                    continue
                if row.get(metric) is not None:
                    by_task[row["task_id"]].append(float(row[metric]))
            summary[metric] = interval([statistics.mean(v) for v in by_task.values()])
            summary[metric]["observations"] = sum(map(len, by_task.values()))
        summary["objective_check_status"] = dict((status, sum(r.get("evaluation_status") == status for r in group))
                                                  for status in sorted({str(r.get("evaluation_status")) for r in group}))
        summary["objective_planned"] = sum(bool(r.get("check_kind")) and
            (args.execute_code or r["check_kind"] != "code") for r in group)
        # Corpus BLEU is computed separately for each repeat, on complete quality datasets only.
        if manifest["suite"] in ("quality", "smoke"):
            import sacrebleu
            by_repeat = defaultdict(list)
            for row in group:
                by_repeat[row["repeat"]].append(row)
            corpus = []
            for repetition, rs in by_repeat.items():
                if all(r["status"] == "ok" for r in rs):
                    corpus.append(sacrebleu.corpus_bleu([r["answer"] for r in rs],
                        [[r["reference"] for r in rs]], tokenize="13a", smooth_method="exp").score)
            summary["corpus_bleu_by_repeat"] = corpus
        summaries.append(summary)
    speed_groups = defaultdict(list)
    for row in records:
        if row["status"] == "ok":
            speed_groups[(row["model"], row["temperature"], row["context"], row["budget"], row["task_id"])].append(row)
    speed = []
    for (model, temperature, context, budget, task_id), rs in speed_groups.items():
        row = dict(model=model, temperature=temperature, context=context, budget=budget, task_id=task_id, n=len(rs))
        row["planned"] = sum(r["model"] == model and r["temperature"] == temperature and r["context"] == context and r["budget"] == budget
                             and r["task_id"] == task_id for r in records)
        for key in ("ttft_sec", "total_latency_sec", "server_tpot_ms", "tokens", "prompt_tokens"):
            measured = interval([r[key] for r in rs if r.get(key) is not None])
            row.update({f"{key}_{k}": measured[k] for k in ("mean", "median", "p95", "ci_low", "ci_high")})
        row["budget_reached_fraction"] = statistics.mean(r.get("truncated", False) for r in rs)
        speed.append(row)
    write_csv(output / "speed_by_condition.csv", speed)
    # Paired differences use task means at identical settings and seed/repeat indices.
    paired = []
    models = manifest["identity"]["settings"]["models"]
    if len(models) == 2 and manifest["suite"] == "quality":
        for temperature in sorted({r["temperature"] for r in records}):
            for metric in ("task_pass", "rougeL_f1", "sentence_bleu"):
                cells = defaultdict(dict)
                for row in records:
                    if metric == "task_pass" and row.get("check_kind") == "code" and not args.execute_code:
                        continue
                    if row["temperature"] == temperature and row.get(metric) is not None:
                        cells[(row["task_id"], row["repeat"])][row["model"]] = float(row[metric])
                diffs = defaultdict(list)
                for (task_id, repeat), cell in cells.items():
                    if all(m in cell for m in models):
                        diffs[task_id].append(cell[models[1]] - cell[models[0]])
                paired.append(dict(metric=metric, temperature=temperature,
                    direction=f"{models[1]} minus {models[0]}",
                    **interval([statistics.mean(v) for v in diffs.values()])))
    result = dict(schema_version=2, analyzed_utc=now(), suite=manifest["suite"],
                  raw_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                  execute_code=args.execute_code, docker_image=args.docker_image if args.execute_code else None,
                  planned=len(plan), received=sum(r["status"] == "ok" for r in records),
                  bootstrap="2000 resamples; quality clustered by task; latency request-level within condition",
                  summaries=summaries, paired_quality=paired,
                  caveats=["Small diagnostic set, not a general model ranking.",
                           "Error and missing counts are reported; no synthetic or imputed measurements.",
                           "Repeated prompts may benefit from KV caching; inspect raw counters.",
                           "Timing groups pooled across input lengths are descriptive only; use speed_by_condition.csv.",
                           "Client TPOT uses fragment arrival times and exact output count; it is not per-token instrumentation.",
                           "Code checks are absent when execute_code=false; inspect coverage before comparing task_pass."])
    write_json(output / "summary.json", result)
    flat = []
    for summary in summaries:
        row = {k: v for k, v in summary.items() if not isinstance(v, (list, dict))}
        for key, value in summary.items():
            if isinstance(value, dict) and "mean" in value:
                row.update({f"{key}_{k}": v for k, v in value.items()})
        flat.append(row)
    write_csv(output / "summary.csv", flat)
    create_figures(output, records, summaries, manifest["suite"], args.execute_code)
    report = ["# Результаты эксперимента", "", f"Серия: {manifest['suite']}",
              f"Успешно получено: {result['received']}/{len(plan)}.",
              "", "Числа и доверительные интервалы: summary.csv. Подробные ответы: responses.csv.",
              "Скорость по отдельным условиям: speed_by_condition.csv.",
              "", "Ошибки и пропуски входят в coverage; условия эксперимента не объединяйте в одну оценку быстродействия.",
              "Ни один график не является основанием для вывода без проверки числа задач и ошибок.",
              "Функциональная проверка кода: " + ("включена" if args.execute_code else "НЕ выполнена"),
              "", "Это автоматический отчёт, а не переписанная статья. Статья обновляется после анализа данных."]
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Offline analysis saved to {output}")


def create_figures(output, records, summaries, suite, execute_code):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.bbox": "tight", "pdf.fonttype": 42})
    models = list(dict.fromkeys(r["model"] for r in records))
    colors = dict(zip(models, ["#2563eb", "#e87924", "#15997c", "#9253b4"]))
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    def save(fig, name):
        for extension in ("png", "pdf"):
            fig.savefig(figures / f"{name}.{extension}", dpi=220)
        plt.close(fig)
    if suite in ("quality", "smoke"):
        for temperature in sorted({r["temperature"] for r in records}):
            selected = [r for r in records if r["temperature"] == temperature]
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            for j, metric in enumerate(("rougeL_f1", "task_pass")):
                for i, model in enumerate(models):
                    values = defaultdict(list)
                    for row in selected:
                        if metric == "task_pass" and row.get("check_kind") == "code" and not execute_code:
                            continue
                        if row["model"] == model and row.get(metric) is not None:
                            values[row["task_id"]].append(float(row[metric]))
                    means = [statistics.mean(v) for v in values.values()]
                    if not means:
                        continue
                    stats = interval(means)
                    scale = 100 if metric == "task_pass" else 1
                    axes[j].bar(i, stats["mean"] * scale, color=colors[model], width=.55)
                    axes[j].errorbar(i, stats["mean"] * scale,
                        yerr=[[(stats["mean"] - stats["ci_low"]) * scale],
                              [(stats["ci_high"] - stats["mean"]) * scale]], color="#222222", capsize=5)
                    axes[j].text(i, 103, f"n={len(means)} tasks", ha="center", fontsize=8)
                axes[j].set_xticks(range(len(models)), models, rotation=12)
                axes[j].set_ylim(0, 112)
                axes[j].set_ylabel("%")
                axes[j].set_title("ROUGE-L F1" if metric == "rougeL_f1" else "Objective task success")
                axes[j].grid(axis="y", alpha=.15)
            fig.suptitle(f"Quality at temperature={temperature}; 95% task bootstrap CI")
            fig.tight_layout()
            save(fig, f"quality_t{temperature:g}")
        categories = list(dict.fromkeys(r["category"] for r in records))
        fig, ax = plt.subplots(figsize=(9, 4.8))
        for i, model in enumerate(models):
            values = []
            for category in categories:
                cells = [r["rougeL_f1"] for r in records if r["model"] == model and r["category"] == category
                         and r["temperature"] == min(x["temperature"] for x in records) and "rougeL_f1" in r]
                values.append(statistics.mean(cells) if cells else float("nan"))
            ax.barh([j + (i - .5) * .36 for j in range(len(categories))], values,
                    height=.34, color=colors[model], label=model)
        ax.set_yticks(range(len(categories)), categories)
        ax.set_xlim(0, 100)
        ax.set_xlabel("ROUGE-L F1, % (lexical overlap; not semantic correctness)")
        ax.legend(loc="lower right")
        ax.grid(axis="x", alpha=.15)
        save(fig, "quality_by_category")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for model in models:
            valid = [r for r in records if r["model"] == model and r["status"] == "ok"]
            for ax, metric in zip(axes, ("ttft_sec", "server_tpot_ms")):
                points = [(r["prompt_tokens"], r[metric]) for r in valid if r.get("prompt_tokens") is not None and r.get(metric) is not None]
                if points:
                    ax.scatter([p[0] for p in points], [p[1] for p in points],
                               s=28, alpha=.6, label=model, color=colors[model])
                ax.set_xlabel("Measured input tokens")
                ax.set_ylabel("Client TTFT, s" if metric == "ttft_sec" else "Server duration / output token, ms")
                ax.grid(alpha=.15)
                if points:
                    ax.legend()
        fig.suptitle("Individual observations; contexts and output budgets are in the CSV")
        fig.tight_layout()
        save(fig, "timing_observations")
