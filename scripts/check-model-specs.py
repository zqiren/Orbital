#!/usr/bin/env python3
"""Advisory drift check: bundled model specs vs. the models.dev catalog.

Compares every catalogued model's ``context_window`` and ``max_output`` in
agent_os/config/providers.json with https://models.dev/api.json and prints,
per provider:

  - context-window mismatches, marked "ours LOWER" (we waste window) or
    "ours HIGHER" (we may send more than the provider accepts — review these);
  - max-output mismatches, same marking;
  - catalogued models the reference does not list.

Differences under 6% (1,000,000 vs 1,048,576 and the like) are hidden unless
--all is given. When the reference lists an input cap below the total context
(e.g. 256K context / 192K input), our window is compared with the input cap,
since that is the largest prompt the provider accepts.

models.dev is third-party. Treat a row as a lead to check against the
provider's own docs or model list, not as the truth. Rows on Anthropic-protocol
entries are tagged [max_tokens]: there max_output is sent as max_tokens and is
deliberately held below the real cap so the prompt plus reply fits the window.

  python3 scripts/check-model-specs.py                      # fetch live
  python3 scripts/check-model-specs.py --snapshot api.json  # offline
  python3 scripts/check-model-specs.py --provider opencode-go --all

Exits 0 regardless of findings, unless --strict, which exits 1 when any
context window is "ours HIGHER". Not a CI gate. Stdlib only.
"""

import argparse
import json
import os
import sys
import urllib.request

MODELS_DEV_URL = "https://models.dev/api.json"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CATALOG = os.path.join(REPO, "agent_os", "config", "providers.json")

# Our provider key -> models.dev provider ids, earlier ids win on conflicts.
PROVIDER_MAP = {
    "openai": ["openai"],
    "anthropic": ["anthropic"],
    "deepseek": ["deepseek"],
    "moonshot": ["moonshotai", "moonshotai-cn"],
    "zhipu": ["zai", "zhipuai"],
    "qwen": ["alibaba", "alibaba-cn"],
    "xai": ["xai"],
    "google": ["google"],
    "mistral": ["mistral"],
    "groq": ["groq"],
    "together": ["togetherai"],
    "openrouter": ["openrouter"],
    "minimax": ["minimax", "minimax-cn"],
    "hunyuan": ["tencent-tokenhub"],
    "opencode-zen": ["opencode"],
    "opencode-go": ["opencode-go"],
}


def load_reference(snapshot):
    if snapshot:
        with open(snapshot, encoding="utf-8") as f:
            return json.load(f)
    req = urllib.request.Request(
        MODELS_DEV_URL, headers={"User-Agent": "model-spec-check/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def reference_models(ref, ids):
    out = {}
    for pid in ids:
        for mid, m in (ref.get(pid) or {}).get("models", {}).items():
            out.setdefault(mid.lower(), m)
    return out


def differs(ours, theirs, threshold):
    if ours == theirs:
        return False
    if ours is None or theirs is None:
        return True
    return abs(ours - theirs) / max(ours, theirs) >= threshold


def direction(ours, theirs):
    if ours is None:
        return "ours MISSING"
    return ("ours LOWER" if ours < theirs else "ours HIGHER").ljust(12)


def fmt(n):
    return "null" if n is None else f"{n:,}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--snapshot", help="read models.dev JSON from this file instead of fetching")
    ap.add_argument("--catalog", default=DEFAULT_CATALOG, help="providers.json to check")
    ap.add_argument("--provider", action="append", help="only these provider keys (repeatable)")
    ap.add_argument("--all", action="store_true", help="show differences under the 6%% threshold too")
    ap.add_argument("--strict", action="store_true", help="exit 1 if any context window is ours HIGHER")
    args = ap.parse_args()

    with open(args.catalog, encoding="utf-8") as f:
        providers = json.load(f)["providers"]
    try:
        ref = load_reference(args.snapshot)
    except Exception as exc:  # advisory: never fail the release on a fetch error
        print(f"could not load models.dev reference: {exc}", file=sys.stderr)
        return 0

    threshold = 0.0 if args.all else 0.06
    higher_ctx = 0

    for pkey, ids in PROVIDER_MAP.items():
        if args.provider and pkey not in args.provider:
            continue
        pdata = providers.get(pkey)
        if not pdata:
            continue
        refm = reference_models(ref, ids)
        ctx_rows, out_rows, absent = [], [], []
        for mid, entry in pdata.get("models", {}).items():
            if mid.startswith("_"):
                continue
            r = refm.get(mid.lower())
            if r is None:
                absent.append(mid)
                continue
            lim = r.get("limit") or {}
            r_ctx, r_out, r_in = lim.get("context"), lim.get("output"), lim.get("input")
            if r_in and r_ctx and r_in < r_ctx:
                r_ctx_cmp, ctx_note = r_in, f" (input cap; total {fmt(r_ctx)})"
            else:
                r_ctx_cmp, ctx_note = r_ctx, ""
            tag = " [max_tokens]" if (entry.get("sdk") or pdata.get("sdk")) == "anthropic" else ""
            o_ctx, o_out = entry.get("context_window"), entry.get("max_output")
            if r_ctx_cmp is not None and differs(o_ctx, r_ctx_cmp, threshold):
                d = direction(o_ctx, r_ctx_cmp)
                higher_ctx += d.startswith("ours HIGHER")
                ctx_rows.append(f"  {d}  {mid:34} ctx {fmt(o_ctx):>11} vs {fmt(r_ctx_cmp):>11}{ctx_note}")
            if r_out is not None and differs(o_out, r_out, threshold):
                out_rows.append(f"  {direction(o_out, r_out)}  {mid:34} out {fmt(o_out):>11} vs {fmt(r_out):>11}{tag}")
        if not (ctx_rows or out_rows or absent):
            continue
        print(f"\n== {pkey}  (models.dev: {', '.join(ids)})")
        if ctx_rows:
            print(" context window:")
            print("\n".join(sorted(ctx_rows)))
        if out_rows:
            print(" max output:")
            print("\n".join(sorted(out_rows)))
        if absent:
            print(" not in models.dev:", ", ".join(absent))

    print(f"\n{higher_ctx} context window(s) where ours is HIGHER than models.dev; "
          "check each against the provider's own docs before release.")
    return 1 if (args.strict and higher_ctx) else 0


if __name__ == "__main__":
    sys.exit(main())
