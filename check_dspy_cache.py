"""Проверка: кэширует ли DSPy ответы при оценке как в туториале GEPA AIME.

Запуск:
    python check_dspy_cache.py               # N=5, только конфигурация туториала (1 платный вызов)
    python check_dspy_cache.py --control     # плюс контроль с cache=False (ещё N платных вызовов)
"""

import argparse
import hashlib
import tempfile
import time

import dspy
from datasets import load_dataset
from dotenv import load_dotenv

TASK_LM = "openrouter/openai/gpt-4.1-mini"


class GenerateResponse(dspy.Signature):
    """Solve the problem and provide the answer in the correct format."""
    problem = dspy.InputField()
    answer = dspy.OutputField()


def metric(example, prediction, trace=None):
    try:
        return int(int(example["answer"]) == int(prediction.answer))
    except (ValueError, TypeError, AttributeError):
        return 0


def run(lm, example, n, threads):
    lm.history.clear()
    program = dspy.ChainOfThought(GenerateResponse)
    evaluate = dspy.Evaluate(devset=[example] * n, metric=metric, num_threads=threads, display_progress=False)
    t0 = time.time()
    with dspy.context(lm=lm):
        evaluate(program)
    elapsed = time.time() - t0

    print(f"{'#':>2}  {'из кэша':>7}  {'токенов':>7}  hash ответа")
    hashes = []
    for i, entry in enumerate(lm.history):
        text = str(entry["outputs"])
        h = hashlib.sha1(text.encode()).hexdigest()[:10]
        hashes.append(h)
        # у ответа из кэша DSPy usage пустой: запроса к API не было
        tokens = entry["usage"].get("completion_tokens")
        print(f"{i:>2}  {'да' if tokens is None else 'нет':>7}  {tokens or '-':>7}  {h}")
    print(f"вызовов LM: {len(lm.history)}, различных ответов: {len(set(hashes))}, время: {elapsed:.1f} c")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="сколько раз повторить задачу")
    parser.add_argument("--threads", type=int, default=1, help="num_threads в dspy.Evaluate")
    parser.add_argument("--problem", type=int, default=0, help="индекс задачи в AIME 2025")
    parser.add_argument("--control", action="store_true", help="дополнительно прогнать с cache=False")
    args = parser.parse_args()

    load_dotenv()
    dspy.configure_cache(disk_cache_dir=tempfile.mkdtemp(prefix="dspy_cache_check_"))

    x = load_dataset("MathArena/aime_2025")["train"][args.problem]
    example = dspy.Example({"problem": x["problem"], "answer": x["answer"]}).with_inputs("problem")

    print(f"== Туториал: temperature=1, кэш по умолчанию, задача {args.problem} × {args.n}")
    run(dspy.LM(TASK_LM, temperature=1.0, max_tokens=32000), example, args.n, args.threads)
    print()

    if args.control:
        print(f"== Контроль: cache=False, задача {args.problem} × {args.n}")
        run(dspy.LM(TASK_LM, temperature=1.0, max_tokens=32000, cache=False), example, args.n, args.threads)


if __name__ == "__main__":
    main()
