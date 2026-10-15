"""Commands are documented in RUN_EXPERIMENTS.md. No work runs on import."""
import argparse
import json
import math
import sys
from pathlib import Path


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return number


def nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return number


def main():
    parser = argparse.ArgumentParser(description="Reproducible Ollama experiments, schema v2")
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight", help="Inspect installation and hardware; no generation")
    pre.add_argument("--models", nargs="+", default=["llama3.2:3b", "qwen2.5:3b"])
    pre.add_argument("--url", default="http://localhost:11434")
    pre.add_argument("--out", default="preflight.json")
    run = sub.add_parser("run", help="Generate new measurements; appends raw responses")
    run.add_argument("--suite", choices=["smoke", "quality", "speed", "cold"], required=True)
    run.add_argument("--models", nargs=2, default=["llama3.2:3b", "qwen2.5:3b"])
    run.add_argument("--url", default="http://localhost:11434")
    run.add_argument("--out", required=True)
    run.add_argument("--repeats", type=positive, default=3)
    run.add_argument("--seed", type=nonnegative, default=1729)
    run.add_argument("--temperatures", nargs="+", type=float, default=[0.0, 0.7])
    run.add_argument("--context", type=positive, default=2048)
    run.add_argument("--budget", type=positive, default=384)
    run.add_argument("--contexts", nargs="+", type=positive, default=[2048, 4096])
    run.add_argument("--input-repetitions", nargs="+", type=positive, default=[1, 8, 24])
    run.add_argument("--budgets", nargs="+", type=positive, default=[64, 256])
    run.add_argument("--timeout", type=positive, default=300)
    run.add_argument("--require-quantization", default="Q4_K_M")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--retry-errors", action="store_true")
    analyze = sub.add_parser("analyze", help="Offline metrics, functional checks and figures")
    analyze.add_argument("--run", required=True)
    analyze.add_argument("--execute-code", action="store_true", help="Run generated code inside isolated Docker only")
    analyze.add_argument("--docker-image", default="python:3.11-slim")
    judge = sub.add_parser("judge", help="Optional third-family judge, both answer orders")
    judge.add_argument("--run", required=True)
    judge.add_argument("--model", required=True)
    judge.add_argument("--url", default="http://localhost:11434")
    judge.add_argument("--temperature", type=float, default=0.0, help="Candidate temperature to evaluate")
    judge.add_argument("--repeat", type=nonnegative, default=0)
    judge.add_argument("--seed", type=nonnegative, default=1729)
    judge.add_argument("--context", type=positive, default=4096)
    judge.add_argument("--timeout", type=positive, default=300)
    human = sub.add_parser("export-human", help="Create blinded CSV and separate private model mapping")
    human.add_argument("--run", required=True)
    human.add_argument("--out", required=True)
    human.add_argument("--temperature", type=float, default=0.0)
    human.add_argument("--repeat", type=nonnegative, default=0)
    human.add_argument("--seed", type=nonnegative, default=9182)
    score = sub.add_parser("score-human", help="Validate and summarize completed human ratings")
    score.add_argument("--ratings", nargs="+", required=True)
    score.add_argument("--key", required=True)
    score.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        from experiments.client import Ollama
        from experiments.runner import preflight, write_json
        result = preflight(Ollama(args.url), args.models)
        write_json(Path(args.out), result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "run":
        if len(set(args.models)) != 2:
            parser.error("Select two different candidate models")
        if any(not math.isfinite(t) or t < 0 or t > 2 for t in args.temperatures):
            parser.error("Temperatures must lie in [0, 2]")
        if len(set(args.temperatures)) != len(args.temperatures):
            parser.error("Duplicate temperatures are not allowed")
        for values in (args.contexts, args.input_repetitions, args.budgets):
            if len(set(values)) != len(values):
                parser.error("Duplicate experiment conditions are not allowed")
        from experiments.runner import run as run_suite
        run_suite(args)
    elif args.command == "analyze":
        from experiments.analysis import analyze as analyze_suite
        analyze_suite(args)
    else:
        from experiments import judging
        {"judge": judging.judge, "export-human": judging.export_human,
         "score-human": judging.score_human}[args.command](args)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
