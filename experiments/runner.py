"""Append-only experiments with explicit plans, metadata and resumability."""
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

from .client import Ollama
from .dataset import TASKS, dataset_hash, speed_task


def now():
    return datetime.now(timezone.utc).isoformat()


def command_output(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=15)
        return {"returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path, data):
    with Path(path).open("a", encoding="utf-8") as file:
        file.write(json.dumps(data, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    # Fail on corruption instead of silently losing observations.
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def source_hash():
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    paths = sorted(root.glob("*.py")) + [root.parent / "research.py"]
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def preflight(client, models):
    tags = client.api("/api/tags")
    installed = {m["name"]: m for m in tags.get("models", [])}
    missing = set(models) - installed.keys()
    if missing:
        raise ValueError(f"Models missing: {sorted(missing)}. Run ollama pull for these exact tags.")
    info = {}
    for model in models:
        show = client.api("/api/show", {"model": model})
        info[model] = dict(tag=installed[model], show=show)
    versions = {}
    for package in ["requests", "sacrebleu", "matplotlib", "numpy", "streamlit"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    try:
        ram_bytes = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        ram_bytes = None
    return dict(created_utc=now(), ollama=client.api("/api/version"),
                models=info, loaded_models=client.api("/api/ps"),
                platform=platform.platform(), machine=platform.machine(),
                processor=platform.processor(), cpu_count=os.cpu_count(),
                ram_bytes=ram_bytes,
                python=sys.version, packages=versions,
                gpu=command_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                    "--format=csv,noheader"]),
                git=command_output(["git", "-C", str(Path(__file__).resolve().parent.parent),
                                    "rev-parse", "HEAD"]),
                source_sha256=source_hash(), dataset_sha256=dataset_hash())


class GpuMonitor:
    def __init__(self, path):
        self.path = path
        self.stop = threading.Event()
        self.thread = None

    def sample(self):
        while not self.stop.is_set():
            stats = command_output(["nvidia-smi",
                "--query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits"])
            append_jsonl(self.path, {"utc": now(), **stats})
            if "error" in stats or stats.get("returncode"):
                break
            self.stop.wait(1)

    def __enter__(self):
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=16)


def options(args, seed, temperature, context, budget):
    return dict(seed=seed, temperature=temperature, num_ctx=context, num_predict=budget,
                top_p=1.0, top_k=0, repeat_penalty=1.0)


def build_plan(args):
    jobs = []
    for repeat in range(args.repeats):
        # Counterbalance which model starts each repeat, while retaining model blocks for 4 GB GPUs.
        models = args.models if repeat % 2 == 0 else list(reversed(args.models))
        for model in models:
            block = []
            if args.suite in ("quality", "smoke"):
                items = TASKS if args.suite == "quality" else [TASKS[9], TASKS[21]]
                for temperature in args.temperatures:
                    for item in items:
                        block.append(dict(model=model, repeat=repeat, task=item,
                            temperature=temperature, context=args.context, budget=args.budget,
                            seed=args.seed + repeat, cold=False))
            elif args.suite == "speed":
                for context in args.contexts:
                    for length in args.input_repetitions:
                        for budget in args.budgets:
                            block.append(dict(model=model, repeat=repeat, task=speed_task(length),
                                temperature=0.0, context=context, budget=budget,
                                seed=args.seed + repeat, cold=False))
            elif args.suite == "cold":
                block.append(dict(model=model, repeat=repeat, task=TASKS[9],
                    temperature=0.0, context=args.context, budget=64,
                    seed=args.seed + repeat, cold=True))
            random.Random(args.seed + repeat).shuffle(block)
            jobs.extend(block)
    for job in jobs:
        encoded = json.dumps(job, sort_keys=True, ensure_ascii=False).encode()
        job["job_id"] = hashlib.sha256(encoded).hexdigest()[:20]
    return jobs


def run(args):
    directory = Path(args.out).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    client = Ollama(args.url, args.timeout)
    metadata = preflight(client, args.models)
    for model in args.models:
        quantization = metadata["models"][model]["show"].get("details", {}).get("quantization_level")
        if args.require_quantization and quantization != args.require_quantization:
            raise ValueError(f"{model}: quantization={quantization}; expected {args.require_quantization}")
    if len({metadata["models"][m]["tag"]["digest"] for m in args.models}) != 2:
        raise ValueError("The selected tags refer to the same weights; select two different models")
    unrelated = [m.get("name") for m in metadata["loaded_models"].get("models", [])
                 if m.get("name") not in args.models]
    if unrelated:
        raise ValueError(f"Other models are loaded: {unrelated}. Stop them before benchmarking.")
    jobs = build_plan(args)
    settings = {k: v for k, v in vars(args).items() if k not in ("out", "resume", "retry_errors", "handler")}
    identity = dict(settings=settings, dataset_sha256=dataset_hash(), source_sha256=source_hash(),
                    model_digests={m: metadata["models"][m]["tag"]["digest"] for m in args.models},
                    ollama_version=metadata["ollama"], packages=metadata["packages"],
                    hardware={k: metadata[k] for k in ("platform", "machine", "processor", "cpu_count", "ram_bytes", "gpu")})
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        if not args.resume:
            raise ValueError("Output directory already contains a run. Use a new --out or --resume.")
        manifest = json.loads(manifest_path.read_text())
        if manifest["identity"] != identity:
            raise ValueError("Resume refused: settings, code, dataset, dependencies or model digests changed.")
    else:
        if (directory / "raw.jsonl").exists():
            raise ValueError("raw.jsonl exists without manifest; select a new output directory")
        manifest = dict(schema_version=2, suite=args.suite, identity=identity,
                        environment=metadata, planned_requests=len(jobs), started_utc=now(),
                        argv=sys.argv, state="running")
        write_json(manifest_path, manifest)
        write_json(directory / "plan.json", jobs)
        write_json(directory / "dataset.json", TASKS)
    previous = read_jsonl(directory / "raw.jsonl")
    latest = {r["job_id"]: r for r in previous}
    completed = {key for key, row in latest.items() if row["status"] == "ok" or not args.retry_errors}
    active = None
    failures = 0
    try:
        with GpuMonitor(directory / "gpu.jsonl"):
            for index, job in enumerate(jobs, 1):
                if job["job_id"] in completed:
                    continue
                row = dict(schema_version=2, suite=args.suite, started_utc=now(), **job)
                model = job["model"]
                opt = options(args, job["seed"], job["temperature"], job["context"], job["budget"])
                try:
                    if active != (model, job["context"]) or job["cold"]:
                        # Clear only the models explicitly selected for this experiment.
                        for candidate in args.models:
                            client.unload(candidate)
                        active = None
                        if not job["cold"]:
                            warm = client.chat(model, "Ответь одним словом: готов.",
                                {**opt, "num_predict": 16})
                            append_jsonl(directory / "warmup.jsonl", dict(model=model,
                                context=job["context"], utc=now(), response=warm))
                            active = (model, job["context"])
                    row.update(status="ok", response=client.chat(model, job["task"]["prompt"], opt))
                    row["loaded_after"] = client.api("/api/ps")
                except Exception as exc:
                    row.update(status="error", error_type=type(exc).__name__, error=str(exc))
                    active = None
                    failures += 1
                row["finished_utc"] = now()
                append_jsonl(directory / "raw.jsonl", row)
                latest[job["job_id"]] = row
                print(f"[{index}/{len(jobs)}] {model} {job['task']['id']} repeat={job['repeat']} {row['status']}", flush=True)
        successful = sum(r["status"] == "ok" for r in latest.values())
        manifest.update(state="complete" if successful == len(jobs) else "completed_with_errors",
                        successful_requests=successful, failed_requests=len(jobs) - successful,
                        finished_utc=now(), gpu_after=command_output(["nvidia-smi"]))
        write_json(manifest_path, manifest)
    except BaseException:
        manifest.update(state="interrupted", interrupted_utc=now())
        write_json(manifest_path, manifest)
        raise
    if failures or manifest["state"] != "complete":
        raise RuntimeError("Some requests failed; inspect raw.jsonl and resume with --retry-errors")
