#!/usr/bin/env python3
"""Build a curated, structured OpenCodeInstruct subset from the public nvidia
source — reproducibly, so the artifact is auditable and regenerable.

Pipeline (nvidia/OpenCodeInstruct -> curated parquet):
  1. per-test filter: keep only the tests the reference solution PASSES
     (`tests_execution_status == "pass"`, aligned with `unit_tests`), and drop
     the TEST (not the whole problem) when the reference fails it. A test a
     plausible solution passes is clean+deterministic (measured 99.5%
     reproducible in a clean sandbox); failing tests are ambiguous so we drop
     them. This keeps problems with a slightly-buggy reference (~2x the yield
     vs requiring average_test_score == 1.0).
  2. parse each raw `assert f(args) == expected` into a structured case
     {entry:{kind:function,name}, args, kwargs, expected, compare:exact}.
     Anything that isn't a clean function-call exact-compare on literals
     (stdin, OR-chains / multi-answer, complex expressions, methods) is dropped.
  3. dedup by id; keep a row only if it has >= --min-cases structured cases.
  4. write parquet with SMALL row-groups (--row-group-size, default 1000) so the
     virtual lazy-loader can fetch a single window slice without pulling a shard.

Output schema matches the existing grader contract: columns
`input`, `structured_cases` (JSON string), `id`.

Usage:
  # validate on a sample (no write needed), with faithfulness exec-check:
  python scripts/build_opencode_curated.py --limit 8000 --validate --dry-run
  # full build to a local parquet:
  python scripts/build_opencode_curated.py --output data/opencode_curated.parquet
  # then push (separate, explicit step):
  python scripts/build_opencode_curated.py --output ... --push-repo R0mAI/opencodeinstruct-curated
"""
from __future__ import annotations

import argparse
import ast
import json
import sys

SRC_REPO = "nvidia/OpenCodeInstruct"
READ_COLUMNS = ("id", "input", "output", "unit_tests", "average_test_score")


def parse_assert(test_src: str) -> dict | None:
    """Parse `assert f(<literals>) == <literal>` into a structured case.

    Returns None for anything that isn't a single function call compared for
    equality against a literal (so stdin, OR-chains, method/attribute calls,
    and non-literal args/expected are all rejected — exactly the cases that
    can't be graded deterministically by a function call).
    """
    try:
        tree = ast.parse(test_src.strip())
    except SyntaxError:
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assert):
        return None
    cmp = tree.body[0].test
    if (
        not isinstance(cmp, ast.Compare)
        or len(cmp.ops) != 1
        or not isinstance(cmp.ops[0], ast.Eq)
    ):
        return None
    call, rhs = cmp.left, cmp.comparators[0]
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
        return None  # only bare function calls (kind=function); skip methods
    try:
        expected = ast.literal_eval(rhs)
        args = [ast.literal_eval(a) for a in call.args]
        kwargs = {}
        for kw in call.keywords:
            if kw.arg is None:
                return None  # **kwargs unpacking
            kwargs[kw.arg] = ast.literal_eval(kw.value)
    except (ValueError, SyntaxError, TypeError):
        return None  # non-literal arg / expected -> not cleanly structurable
    case = {
        "entry": {"kind": "function", "name": call.func.id},
        "args": args,
        "kwargs": kwargs,
        "expected": expected,
        "compare": "exact",
    }
    try:
        canonical = json.loads(json.dumps(case))
    except (TypeError, ValueError):
        return None  # non-serializable (set, bytes, ...) -> can't reach grader
    if canonical != case:
        # Lossy JSON round-trip (e.g. a tuple arg/expected became a list): the
        # grader would compare against a different value than the reference
        # validated, so drop it rather than inject a false negative.
        return None
    return canonical


def _coerce_tests(raw) -> list:
    """nvidia streams `unit_tests` as a JSON-encoded string; decode it.
    Falls back to native list/array forms.
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            v = json.loads(raw)
            return list(v) if isinstance(v, (list, tuple)) else []
        except json.JSONDecodeError:
            return []
    try:
        return list(raw)
    except TypeError:
        return []


def iter_curated(limit: int, min_cases: int):
    """Yield curated rows {id, input, structured_cases(list)} from nvidia."""
    from datasets import load_dataset

    ds = load_dataset(SRC_REPO, split="train", streaming=True)
    seen: set[str] = set()
    stats = {"seen": 0, "kept": 0, "dup": 0, "align_skip": 0}
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        stats["seen"] += 1
        # Per-test curation: nvidia gives a pass/fail status per test
        # (`tests_execution_status`, aligned with `unit_tests`). Keep only the
        # tests the reference solution PASSES — a test a plausible solution
        # passes is a clean, deterministic one (measured 99.5% reproducible in a
        # clean sandbox). Tests the reference fails are ambiguous (bad test OR
        # bad reference), so drop the TEST, not the whole problem — this keeps
        # problems whose reference is only slightly buggy (~2x the yield).
        uts = _coerce_tests(row.get("unit_tests"))
        sts = _coerce_tests(row.get("tests_execution_status"))
        if not uts or len(sts) != len(uts):
            stats["align_skip"] += 1
            continue
        rid = row.get("id")
        if rid in seen:
            stats["dup"] += 1
            continue
        cases = []
        for t, st in zip(uts, sts):
            if str(st).lower() != "pass":
                continue
            c = parse_assert(str(t))
            if c is not None:
                cases.append(c)
        if len(cases) < min_cases:
            continue
        seen.add(rid)
        stats["kept"] += 1
        yield {"id": rid, "input": row.get("input", ""), "structured_cases": cases}
    yield {"__stats__": stats}


def faithfulness_check(rows: list[dict], outputs: dict[str, str], sample: int) -> dict:
    """Exec each reference solution and run its structured cases — confirm the
    parse is faithful (cases pass against the reference). Sandboxed by SIGALRM.
    """
    import contextlib
    import io
    import os
    import signal

    class TO(Exception):
        pass

    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TO()))
    devnull = io.StringIO()
    ok = total = err = 0
    for r in rows[:sample]:
        code = _strip_fences(outputs.get(r["id"], ""))
        ns: dict = {}
        try:
            signal.alarm(4)
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                exec(code, ns)
            signal.alarm(0)
        except (Exception, TO):
            signal.alarm(0)
            err += 1
            continue
        for c in r["structured_cases"]:
            fn = ns.get(c["entry"]["name"])
            if not callable(fn):
                continue
            total += 1
            try:
                signal.alarm(4)
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    got = fn(*c["args"], **c["kwargs"])
                signal.alarm(0)
                if got == c["expected"]:
                    ok += 1
            except (Exception, TO):
                signal.alarm(0)
    return {"cases_run": total, "passed": ok, "exec_err": err}


def _strip_fences(out: str) -> str:
    import re

    m = re.search(r"```(?:python)?\s*(.*?)```", out, re.S)
    return m.group(1) if m else out


def _write_streaming(args) -> dict:
    """Memory-bounded full build: stream curated rows and write the parquet in
    row-group-aligned batches, so millions of rows never sit in RAM at once.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([
        ("input", pa.string()),
        ("structured_cases", pa.string()),
        ("id", pa.string()),
    ])
    batch_rows = max(args.row_group_size * 50, args.row_group_size)
    writer = pq.ParquetWriter(args.output, schema)
    batch: list[dict] = []
    stats: dict = {}

    def flush() -> None:
        if not batch:
            return
        tbl = pa.table({
            "input": [r["input"] for r in batch],
            "structured_cases": [json.dumps(r["structured_cases"]) for r in batch],
            "id": [r["id"] for r in batch],
        }, schema=schema)
        writer.write_table(tbl, row_group_size=args.row_group_size)
        batch.clear()

    try:
        for item in iter_curated(args.limit, args.min_cases):
            if "__stats__" in item:
                stats = item["__stats__"]
                break
            batch.append(item)
            if len(batch) >= batch_rows:
                flush()
        flush()
    finally:
        writer.close()
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = full dataset")
    ap.add_argument("--min-cases", type=int, default=3,
                    help="keep a problem only if >= this many ref-passing cases parse")
    ap.add_argument("--row-group-size", type=int, default=1000)
    ap.add_argument("--output", default=None, help="parquet path to write")
    ap.add_argument("--dry-run", action="store_true", help="don't write parquet")
    ap.add_argument("--validate", action="store_true",
                    help="exec references to confirm parsed cases are faithful")
    ap.add_argument("--push-repo", default=None,
                    help="HF dataset repo id to push to (explicit, last step)")
    args = ap.parse_args()

    # Full build: stream-write to parquet with bounded memory, then push.
    if args.output and not args.dry_run and not args.validate:
        stats = _write_streaming(args)
        yp = 100 * stats["kept"] / max(stats["seen"], 1)
        print(f"seen={stats['seen']} kept={stats['kept']} ({yp:.1f}%) "
              f"dup={stats['dup']} align_skip={stats['align_skip']} "
              f"-> {args.output} (row_group_size={args.row_group_size})")
        if args.push_repo:
            from huggingface_hub import HfApi
            HfApi().upload_file(
                path_or_fileobj=args.output,
                path_in_repo="data/train-00000.parquet",
                repo_id=args.push_repo,
                repo_type="dataset",
            )
            print(f"pushed -> {args.push_repo}")
        return 0

    # Sample / validate path: collect to list for reporting (small samples only).
    rows: list[dict] = []
    stats: dict = {}
    outputs: dict[str, str] = {}
    for item in iter_curated(args.limit, args.min_cases):
        if "__stats__" in item:
            stats = item["__stats__"]
            break
        rows.append(item)

    yield_pct = 100 * stats["kept"] / max(stats["seen"], 1)
    avg_cases = (sum(len(r["structured_cases"]) for r in rows) / max(len(rows), 1))
    print(f"seen={stats['seen']} kept={stats['kept']} ({yield_pct:.1f}%) "
          f"dup={stats['dup']} align_skip={stats['align_skip']} "
          f"avg_cases/row={avg_cases:.1f}")

    print("\n-- sample structured_cases (first 2 rows) --")
    for r in rows[:2]:
        print(f"  id={r['id']} input={r['input'][:70]!r}")
        print(f"    cases[0]={json.dumps(r['structured_cases'][0])}")

    if args.validate:
        # need the reference outputs; re-stream the same prefix to grab them
        from datasets import load_dataset
        ids = {r["id"] for r in rows}
        ds = load_dataset(SRC_REPO, split="train", streaming=True)
        for i, row in enumerate(ds):
            if args.limit and i >= args.limit:
                break
            if row.get("id") in ids:
                outputs[row["id"]] = row.get("output", "")
        fc = faithfulness_check(rows, outputs, sample=min(200, len(rows)))
        rate = 100 * fc["passed"] / max(fc["cases_run"], 1)
        print(f"\n-- faithfulness: {fc['passed']}/{fc['cases_run']} parsed cases "
              f"pass vs reference ({rate:.1f}%), exec_err={fc['exec_err']} --")

    if args.output and not args.dry_run:
        import pyarrow as pa
        import pyarrow.parquet as pq
        table = pa.table({
            "input": [r["input"] for r in rows],
            "structured_cases": [json.dumps(r["structured_cases"]) for r in rows],
            "id": [r["id"] for r in rows],
        })
        pq.write_table(table, args.output, row_group_size=args.row_group_size)
        print(f"\nwrote {len(rows)} rows -> {args.output} "
              f"(row_group_size={args.row_group_size})")
        if args.push_repo:
            from huggingface_hub import HfApi
            HfApi().upload_file(
                path_or_fileobj=args.output,
                path_in_repo="data/train-00000.parquet",
                repo_id=args.push_repo,
                repo_type="dataset",
            )
            print(f"pushed -> {args.push_repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
