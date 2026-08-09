#!/usr/bin/env python3
"""Check that gemma-4's vision harness and the driver's forward are still the same code.

`gemma4_vision_full_parity_bf16.cu` does NOT include the driver. It carries its
own `run_gemma4_vision` and its own 13 kernels, and its header says why: the
kernels and the entry point "port directly into the driver module". So the
harness verifies a REFERENCE, and the driver is correct only as long as the
copy in `model/gemma4/gemma4_vision_forward.cu` still matches it.

Nothing enforced that. This does.

Measured when this was written: all 12 shared kernels identical modulo
parameter names, and both entry points issue the same 25 launches in the same
order. The names had already drifted -- the driver takes `k_matmul`, `k_rms`,
`k_add` and `k_clamp` from `tower_naive_kernels.cuh` instead of defining its
own, and `k_addpos`/`k_rope` were renamed to `k_addpos_grid2d`/
`k_rope_axial2d` when four tower names turned out to each mean two things.
Those are recorded in RENAMES below; a new divergence is a finding.

What this does NOT check: the driver reads its shapes from `VisRawWeights`
while the harness hardcodes them (Hd=768, NH=12, IM=3072, TXT=2560 ...). That
generalisation is the driver's, and is deliberate.

    python scripts/gemma4-vision-drift-check.py     # exits non-zero on drift
"""

from __future__ import annotations

import pathlib
import re
import sys

HARNESS = "crates/driver-cuda/csrc/tests/gemma4_vision_full_parity_bf16.cu"
DRIVER = "crates/driver-cuda/csrc/src/model/gemma4/gemma4_vision_forward.cu"
# The driver takes these from the shared tower header rather than defining them.
SHARED = ["crates/driver-cuda/csrc/src/model/tower_naive_kernels.cuh",
          "crates/driver-cuda/csrc/src/model/gemma4/gemma4_naive_kernels.cuh"]

# harness name -> driver name, for the renames that are known and intended.
RENAMES = {
    "k_addpos": "k_addpos_grid2d",
    "k_rope": "k_rope_axial2d",
}

# Kernels whose bodies differ ON PURPOSE, with the reason. A body compare that
# reports a difference everyone already knows about gets ignored, and then the
# next one gets ignored too -- so each entry has to say why, and anything not
# listed is still a finding.
ALLOWED_BODY_DIFF = {
    "k_clamp":
        "interface, not arithmetic: the harness takes the clip bounds as host "
        "floats (`float lo, float hi`) because it read them out of a .npy; the "
        "driver takes device pointers (`const bf* lo, const bf* hi`) because "
        "they arrive as DeviceTensors with the model, and it also tolerates a "
        "null bound as +/-inf. For equal, present bounds the two compute the "
        "same clamp.",
}


def kernel_bodies(path: pathlib.Path) -> dict[str, str]:
    """name -> body, comments and whitespace stripped, identifiers normalised.

    Parameter names are NOT semantics: the driver's shared `k_add` writes
    `a[i] = a[i] + b[i]` where the harness writes `h[i] = h[i] + x[i]`. Those
    are the same kernel and an exact-text compare calls them different, which
    is a false alarm the reader then has to run down by hand.
    """
    s = path.read_text()
    out: dict[str, str] = {}
    for m in re.finditer(r'__global__\s+void\s+(\w+)\s*\(', s):
        # signature, so the parameter names can be mapped to positions
        sig_end = s.find(")", m.end())
        params = s[m.end():sig_end]
        names = re.findall(r'(\w+)\s*(?:\[\s*\])?\s*(?:,|$)', params)
        i = s.find("{", sig_end)
        depth, j = 1, i + 1
        while j < len(s) and depth:
            if s[j] == "{":
                depth += 1
            elif s[j] == "}":
                depth -= 1
            j += 1
        assert depth == 0, f"{path.name}: braces do not balance in {m.group(1)}"
        body = re.sub(r'//[^\n]*', '', s[i:j])
        for k, p in enumerate(names):
            body = re.sub(r'(?<![\w.])' + re.escape(p) + r'(?![\w])', f"@{k}", body)
        out[m.group(1)] = re.sub(r'\s+', '', body)
    return out


def launch_sequence(path: pathlib.Path) -> list[str]:
    s = path.read_text()
    i = s.find("void run_gemma4_vision")
    assert i >= 0, f"{path}: no run_gemma4_vision"
    j = s.find("\n}\n", i)
    body = s[i:j]
    # Strip comments FIRST. Without this a launch commented out with `//` is
    # still counted, and the check reports "same order" for a driver that
    # stopped issuing it -- which is the exact edit someone makes while
    # debugging and forgets to undo. Found by feeding the checker that edit.
    body = re.sub(r'//[^\n]*', '', body)
    body = re.sub(r'/\*.*?\*/', '', body, flags=re.S)
    return re.findall(r'(k_[a-z0-9_]+)<<<', body)


def main() -> int:
    root = pathlib.Path(".")
    h_k = kernel_bodies(root / HARNESS)
    d_k = kernel_bodies(root / DRIVER)
    for extra in SHARED:
        d_k.update(kernel_bodies(root / extra))

    problems: list[str] = []

    for name, body in sorted(h_k.items()):
        target = RENAMES.get(name, name)
        if target not in d_k:
            problems.append(
                f"{name}: the harness has it, the driver has no {target}. "
                f"Either the driver dropped it or it was renamed -- add the "
                f"rename to RENAMES if it was intended.")
        elif d_k[target] != body and name not in ALLOWED_BODY_DIFF:
            problems.append(
                f"{name} -> {target}: bodies differ. The harness's copy is the "
                f"one checked against HF; a driver that computes something "
                f"else is not covered by any test.")

    h_seq = [RENAMES.get(k, k) for k in launch_sequence(root / HARNESS)]
    d_seq = launch_sequence(root / DRIVER)
    if h_seq != d_seq:
        import difflib
        problems.append("run_gemma4_vision issues a different sequence:")
        problems += ["    " + line for line in
                     difflib.unified_diff(h_seq, d_seq, "harness", "driver",
                                          lineterm="", n=1)]

    if problems:
        print("gemma-4 vision: harness and driver have DRIFTED\n")
        for p in problems:
            print(f"  {p}")
        print("\nThe harness does not include the driver, so this drift is "
              "invisible to every build and to the parity run itself.")
        return 1

    print(f"gemma-4 vision: in sync "
          f"({len(h_k)} kernels, {len(h_seq)} launches, same order)")
    for k, why in sorted(ALLOWED_BODY_DIFF.items()):
        print(f"  allowed difference, {k}: {why.split(':')[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
