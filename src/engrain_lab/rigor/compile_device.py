"""What a device does with the work `groups` is made of.

`groups` is 54.4% of a compile: from every lexer state, scan every token of the
vocabulary through the byte lexer and record where it lands. Measured over 40
corpus schemas that is 1,857M independent scans and 11.98G byte steps, and the
host does it in 2,179 ms on 24 cores - 5.50G byte steps a second.

The step is a gather, `next = table[state * 256 + byte]`, dependent along a
token and independent across every (state, token) pair. That is the shape this
measures, as a real kernel rather than a chain of framework operations: a
tensor expression writes the state vector to memory and reads it back at every
byte, and measured that way the device reports 0.9x the host - which is a
measurement of the framework. A kernel keeps the state in a register, and that
is the whole difference.

Tokens are bucketed by length and each bucket is launched with its own length,
so no lane walks past the end of its token. The host stops at the end of a
token too, and anything else would flatter the device.

What this does *not* model: grouping the scanned tokens by how they read, 6% of
the stage on the host, and the settle/restart splice. It measures the 94% that
is the scan.

    python -m engrain_lab.rigor.compile_device --states 131 1931
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS = Path("results")
# What the host does, from `rigor.compile_phases` over 40 corpus schemas.
HOST_GSTEPS = 5.50


def _kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def scan(
        table_ptr,
        bytes_ptr,
        out_ptr,
        rows,
        tokens,
        stride,
        BLOCK: tl.constexpr,
        LENGTH: tl.constexpr,
    ):
        """One lane per (lexer state, token), walking that token's bytes.

        `LENGTH` is constant per launch because the tokens are bucketed by it,
        so the loop is exactly the work the token needs and no lane idles
        behind a longer one. The state stays in a register across the walk,
        which is the thing a tensor expression cannot do.
        """
        start = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = start < rows
        state = (start // tokens).to(tl.int32)
        token = start % tokens
        for at in range(LENGTH):
            byte = tl.load(bytes_ptr + token * stride + at, mask=live, other=0)
            state = tl.load(table_ptr + state * 256 + byte, mask=live, other=0)
        tl.store(out_ptr + start, state, mask=live)

    return scan


def main() -> int:
    import torch
    import triton

    from engrain_lab.rigor.harness import load_vocabulary

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--states",
        type=int,
        nargs="+",
        default=[131, 1931],
        help="lexer states to scan from; the corpus p50 and p90",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cap", type=int, default=32, help="longest token walked")
    arguments = parser.parse_args()

    scan = _kernel()
    vocabulary = load_vocabulary(arguments.model)
    buckets: dict[int, list[bytes]] = {}
    for token in vocabulary:
        if token and len(token) <= arguments.cap:
            buckets.setdefault(len(token), []).append(token)
    held = {
        length: torch.tensor(
            [list(token) for token in tokens], dtype=torch.int32, device="cuda"
        )
        for length, tokens in buckets.items()
    }
    kept = sum(len(tokens) for tokens in buckets.values())
    useful = sum(length * len(tokens) for length, tokens in buckets.items())
    print(
        f"{kept} tokens of {len(vocabulary)} are under {arguments.cap} bytes, "
        f"{useful / 1e6:.2f}M byte steps per lexer state"
    )

    report = []
    for states in arguments.states:
        # A real transition table, in size and dtype. Its contents only decide
        # which line a step reads, and a random table is the pessimistic case:
        # a real lexer's states cluster, so its reads hit warmer lines.
        table = torch.randint(
            0, states, (states * 256,), dtype=torch.int32, device="cuda"
        )
        out = {
            length: torch.empty(
                states * tokens.shape[0], dtype=torch.int32, device="cuda"
            )
            for length, tokens in held.items()
        }

        def once(states=states, table=table, out=out) -> None:
            for length, tokens in held.items():
                rows = states * tokens.shape[0]
                scan[(triton.cdiv(rows, 256),)](
                    table,
                    tokens,
                    out[length],
                    rows,
                    tokens.shape[0],
                    length,
                    BLOCK=256,
                    LENGTH=length,
                    num_warps=4,
                )

        for _ in range(2):
            once()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(arguments.repeats):
            once()
        stop.record()
        torch.cuda.synchronize()
        each = start.elapsed_time(stop) / arguments.repeats
        steps = useful * states
        rate = steps / each / 1e6
        print(
            f"{states:>5} states: {each:8.2f} ms, {steps / 1e9:6.2f}G byte steps, "
            f"{rate:7.2f}G steps/s ({rate / HOST_GSTEPS:5.1f}x the host's "
            f"{HOST_GSTEPS}G on 24 cores)"
        )
        report.append(
            {
                "states": states,
                "ms": each,
                "steps": steps,
                "gsteps_per_s": rate,
                "host_gsteps_per_s": HOST_GSTEPS,
                "speedup": rate / HOST_GSTEPS,
            }
        )
        del table, out
        torch.cuda.empty_cache()

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "compile-device.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
