"""A narrowed mask must not reach a row the engine did not flag.

`overflow` is the whole safety story of a bounded replay: a chain that meets a
ceiling stops, the terminal it was testing goes unadmitted, and the flag says
so, so a caller refills that row from the reference matcher. The mask is exact
either way and the cost is that the row leaves the device.

The flag is per sequence, and two things carry a mask *between* sequences:

  * the memo, which remembers a computed mask under a key describing the parse
    state - so a narrowed row publishes a truncated answer that a later step
    reads as exact, and
  * the dedupe, where sequences in the same state share one computation - so a
    row copying from a narrowed neighbour gets the truncation without the flag.

Both were live. Found by `rigor.online` as a serving row twelve words of bits
short of its own matcher, with nothing in `problems()` to say so, and three of
four hundred requests killed by the token the mask had dropped.

Reaching it needs replays that meet the ceiling, which the argument forces:
`window` here is what a deep enough document reaches on its own.

    python -m engrain_lab.verify.narrowing [schemas] [rows] [steps] [window]
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

INSTANCES = Path("results/corpus-exact.json")


def main() -> None:
    import torch

    import engrain.internals as E
    from transformers import AutoTokenizer

    schemas = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    rows = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    # Small enough that ordinary corpus documents meet it. The engine's own
    # default is sized from the grammars it holds, so nothing meets it until a
    # document nests further than any schema suggested it would.
    window = int(sys.argv[4]) if len(sys.argv) > 4 else 4

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    vocabulary: list[bytes] = []
    for token_id in range(len(tokenizer)):
        piece = tokenizer.convert_ids_to_tokens(token_id)
        vocabulary.append(tokenizer.convert_tokens_to_string([piece]).encode())

    compiler = E.Compiler(vocabulary)
    instances = json.loads(INSTANCES.read_text())["instances"]
    grammars: list[Any] = []
    for instance in instances:
        if len(grammars) >= schemas:
            break
        try:
            grammars.append(compiler.compile_json_schema(instance["schema"]))
        except Exception:  # noqa: BLE001
            continue
    if not grammars:
        raise SystemExit(f"no schema in {INSTANCES} compiled")

    pool = E.DeviceGrammar(max_configs=16, window=window)
    ids = [pool.admit(grammar) for grammar in grammars]
    batch = pool.new_batch(rows)
    rng = random.Random(5)

    def fresh() -> dict:
        which = rng.randrange(len(grammars))
        return {"which": which, "matcher": grammars[which].matcher(0)}

    live = [fresh() for _ in range(rows)]
    flagged = 0
    walked = 0

    for step in range(steps):
        wanted = [ids[entry["which"]] for entry in live]
        batch.set_grammars(wanted)
        try:
            batch.set_matchers([entry["matcher"] for entry in live])
        except (E.ConfigurationsExceeded, E.StackTooDeep):
            # A ceiling the *host* can see refuses before the fill, which is a
            # different mechanism and has its own answer in the backend.
            live = [fresh() for _ in range(rows)]
            continue
        device = batch.fill_mask().to("cpu")
        _, flags = batch.problems()
        narrowed = flags.to("cpu").bool()
        flagged += int(narrowed.sum())
        walked += 1

        for row, entry in enumerate(live):
            if narrowed[row]:
                continue
            reference = torch.zeros(pool.mask_words, dtype=torch.int32)
            entry["matcher"].fill_bitmask(reference)
            if torch.equal(device[row], reference):
                continue
            extra = int(((device[row] & ~reference) != 0).sum())
            missing = int(((reference & ~device[row]) != 0).sum())
            raise SystemExit(
                f"step {step} row {row} under grammar {wanted[row]} disagrees "
                f"with its own matcher in {extra} words of extra bits and "
                f"{missing} of missing, and problems() did not flag it"
            )

        for row, entry in enumerate(live):
            allowed = entry["matcher"].allowed_tokens()
            if not allowed:
                live[row] = fresh()
                continue
            entry["matcher"].accept_token(rng.choice(allowed))

    if not flagged:
        # Without a single narrowed row this check asserted nothing: the window
        # was wide enough that no replay met it, and the two paths it exists to
        # cover were never taken.
        raise SystemExit(
            f"no row met the window of {window} over {walked} steps, so "
            "nothing here was tested - lower it"
        )
    print(
        f"{walked} steps over {len(grammars)} grammars x {rows} rows: "
        f"{flagged} row-steps narrowed and flagged, 0 narrowed silently"
    )


if __name__ == "__main__":
    main()
