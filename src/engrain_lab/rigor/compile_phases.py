"""Where a compile's milliseconds go, stage by stage, over the corpus.

The cold path is the one number this engine loses on by two orders of
magnitude - 227 ms at the median against llguidance's 1.2 - and "make the
compiler faster" is not a plan until it says *which part*. The pipeline
already laps itself behind `ENGRAIN_WHY`; this runs it over real schemas and
adds the laps up, so the answer is a share of a total rather than a comment.

    python -m engrain_lab.rigor.compile_phases --schemas 40
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

RESULTS = Path("results")

# `  Some(level) stage 1.234ms` - the lap line the pipeline writes.
LAP = re.compile(r"^\s+(\S+)\s+(\w+)\s+([0-9.]+)(ns|µs|ms|s)\s*$")
SCALE = {"ns": 1e-6, "µs": 1e-3, "ms": 1.0, "s": 1e3}


def _child(corpus: str, count: int, model: str) -> None:
    """Compile in a subprocess, because the trace goes to stderr."""
    import engrain.internals as E
    from engrain_lab.rigor.harness import load_vocabulary

    vocabulary = load_vocabulary(model)
    compiler = E.Compiler(vocabulary)
    lengths = [len(token) for token in vocabulary if token]
    print(
        f"VOCAB {len(vocabulary)} {sum(lengths) / max(len(lengths), 1):.4f}",
        flush=True,
    )
    instances = json.loads(Path(corpus).read_text())["instances"]
    done = 0
    for instance in instances:
        if done >= count:
            break
        try:
            compiled = compiler.compile_json_schema(instance["schema"])
        except Exception:  # noqa: BLE001
            continue
        done += 1
        print(f"SCHEMA {done} {compiled.num_lexer_states}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--corpus", default=str(RESULTS / "corpus-exact.json"))
    parser.add_argument("--schemas", type=int, default=40)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    arguments = parser.parse_args()

    if arguments.child:
        _child(arguments.corpus, arguments.schemas, arguments.model)
        return 0

    run = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "engrain_lab.rigor.compile_phases",
            "--child",
            "--corpus",
            arguments.corpus,
            "--schemas",
            str(arguments.schemas),
            "--model",
            arguments.model,
        ],
        env={**__import__("os").environ, "ENGRAIN_WHY": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    if run.returncode != 0:
        print(run.stderr[-4000:])
        return 1

    total: dict[str, float] = {}
    attempts: dict[str, int] = {}
    for line in run.stderr.splitlines():
        found = LAP.match(line)
        if not found:
            continue
        _, stage, value, unit = found.groups()
        total[stage] = total.get(stage, 0.0) + float(value) * SCALE[unit]
        attempts[stage] = attempts.get(stage, 0) + 1

    # The shape of the dominant stage, from the same schemas that were timed.
    # `groups` scans the whole vocabulary from every lexer state, so its work
    # is states x tokens x the bytes of a token, and whether a device could
    # take it depends on those three rather than on the milliseconds.
    states = 0
    for line in run.stdout.splitlines():
        if line.startswith("SCHEMA "):
            states += int(line.split()[2])
    tokens = 0
    span = 0.0
    for line in run.stdout.splitlines():
        if line.startswith("VOCAB "):
            tokens, span = int(line.split()[1]), float(line.split()[2])

    compiled = run.stdout.count("SCHEMA ")
    if not compiled or not total:
        print("nothing compiled, or the trace changed shape")
        return 1
    spent = sum(total.values())
    print(f"{compiled} schemas compiled, {spent:.0f} ms of pipeline in total\n")
    print(f"{'stage':<10}{'ms total':>10}{'share':>8}{'ms/schema':>12}{'laps':>7}")
    for stage, held in sorted(total.items(), key=lambda item: -item[1]):
        print(
            f"{stage:<10}{held:>10.0f}{100 * held / spent:>7.1f}%"
            f"{held / compiled:>12.1f}{attempts[stage]:>7}"
        )

    scans = states * tokens
    steps = scans * span
    print(
        f"\ngrouping scanned {states} lexer states x {tokens} tokens = "
        f"{scans / 1e6:.0f}M independent scans, {steps / 1e9:.2f}G byte steps,\n"
        f"in {total.get('groups', 0):.0f} ms on every core: "
        f"{scans / max(total.get('groups', 1e-9), 1e-9) / 1e3:.1f}M scans/s, "
        f"{steps / max(total.get('groups', 1e-9), 1e-9) / 1e6:.2f}G steps/s"
    )

    report = {
        "schemas": compiled,
        "total_ms": spent,
        "lexer_states": states,
        "vocabulary": tokens,
        "mean_token_bytes": span,
        "stages": {
            stage: {
                "ms": held,
                "share": held / spent,
                "ms_per_schema": held / compiled,
                "laps": attempts[stage],
            }
            for stage, held in total.items()
        },
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "compile-phases.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
