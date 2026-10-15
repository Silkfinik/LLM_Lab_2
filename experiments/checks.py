"""Deterministic checks; generated Python executes only inside opt-in Docker."""
import ast
import json
import re
import subprocess
import uuid
from collections import Counter


def words(text):
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


def rouge_f1(prediction, reference, tokenizer=words):
    a, b = tokenizer(prediction), tokenizer(reference)
    overlap = sum((Counter(a) & Counter(b)).values())
    previous = [0] * (len(b) + 1)
    for x in a:
        current = [0]
        for j, y in enumerate(b):
            current.append(previous[j] + 1 if x == y else max(previous[j + 1], current[-1]))
        previous = current
    denom = len(a) + len(b)
    return {"rouge1_f1": 200 * overlap / denom if denom else 0,
            "rougeL_f1": 200 * previous[-1] / denom if denom else 0}


def reference_metrics(text, reference):
    import sacrebleu
    result = rouge_f1(text, reference)
    legacy = rouge_f1(text, reference, lambda s: s.lower().split())
    result.update({"rouge1_whitespace_f1": legacy["rouge1_f1"],
                   "rougeL_whitespace_f1": legacy["rougeL_f1"],
                   "sentence_bleu": sacrebleu.sentence_bleu(text, [reference],
                       smooth_method="exp", tokenize="13a", use_effective_order=True).score})
    return result


def same_typed_value(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(b, dict):
        return a.keys() == b.keys() and all(same_typed_value(a[k], b[k]) for k in b)
    if isinstance(b, list):
        return len(a) == len(b) and all(same_typed_value(x, y) for x, y in zip(a, b))
    return a == b


def no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


DOCKER_HARNESS = r'''
import contextlib, io, json, resource, sys
resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
payload = json.loads(sys.stdin.read())
namespace = {}
try:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        exec(compile(payload['code'], '<candidate>', 'exec'), namespace)
        fn = namespace[payload['function']]
        outcomes = []
        for args, expected in payload['cases']:
            try:
                actual = fn(*args)
                outcomes.append(type(actual) is type(expected) and actual == expected)
            except Exception:
                outcomes.append(False)
    print(json.dumps({'passed': sum(outcomes), 'total': len(outcomes), 'all_pass': all(outcomes)}))
except BaseException as exc:
    print(json.dumps({'passed': 0, 'total': len(payload['cases']), 'all_pass': False, 'error': type(exc).__name__}))
'''


def check_task(item, text, execute_code=False, docker_image="python:3.11-slim"):
    kind = item.get("check")
    result = dict(check_kind=kind, task_pass=None, format_pass=None,
                  functional_pass=None, evaluation_status="human_required")
    if kind == "exact":
        # Full answer, not substring search: wrong extra numbers cannot earn a pass.
        result.update(task_pass=text.strip().casefold() == item["reference"].casefold(),
                      format_pass=text.strip() == item["reference"], evaluation_status="evaluated")
    elif kind == "json":
        try:
            parsed = json.loads(text, object_pairs_hook=no_duplicate_keys,
                                parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
            result.update(format_pass=True, task_pass=same_typed_value(parsed, item["expected"]),
                          evaluation_status="evaluated")
        except (ValueError, TypeError):
            result.update(format_pass=False, task_pass=False, evaluation_status="evaluated")
    elif kind == "code":
        # Strip a single enclosing fence for functional evaluation only. Fences still fail format.
        stripped = text.strip()
        match = re.fullmatch(r"```(?:python)?\s*\n(.*?)\n```", stripped, flags=re.S)
        source = match.group(1) if match else stripped
        try:
            tree = ast.parse(source)
            functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
            target_present = any(n.name == item["function"] for n in functions)
            pure_function = (len(tree.body) == 1 and len(functions) == 1 and target_present
                             and not functions[0].decorator_list
                             and not any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(tree)))
            result["format_pass"] = pure_function and match is None
        except SyntaxError:
            result.update(format_pass=False, functional_pass=False, task_pass=False,
                          evaluation_status="syntax_error")
            return result
        if not execute_code:
            result["evaluation_status"] = "code_not_executed"
            return result
        container_name = "llm-check-" + uuid.uuid4().hex
        command = ["docker", "run", "--rm", "--name=" + container_name, "--pull=never", "--network=none", "--read-only",
                   "--cap-drop=ALL", "--security-opt=no-new-privileges", "--pids-limit=32",
                   "--memory=128m", "--cpus=1", "--user=65534:65534", "-i",
                   docker_image, "python", "-I", "-c", DOCKER_HARNESS]
        try:
            response = subprocess.run(command, input=json.dumps(dict(code=source,
                function=item["function"], cases=item["cases"])), text=True,
                capture_output=True, timeout=10)
            if response.returncode:
                # OOM/killed code and unavailable Docker are distinct from normal failures.
                result.update(evaluation_status="sandbox_error", detail=response.stderr[-1000:])
                return result
            outcome = json.loads(response.stdout)
            result.update(functional_pass=outcome["all_pass"],
                          task_pass=bool(outcome["all_pass"] and result["format_pass"]),
                          evaluation_status="evaluated", code_tests=outcome)
        except subprocess.TimeoutExpired:
            result.update(evaluation_status="sandbox_timeout", functional_pass=False, task_pass=False)
        except (OSError, ValueError, KeyError) as exc:
            result.update(evaluation_status="sandbox_error", detail=str(exc))
        finally:
            # Killing a timed-out Docker client alone does not stop its container.
            try:
                subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                pass
    return result
