#!/usr/bin/env python3
"""
Classify and analyze random 5000 samples from nvidia/OpenMathInstruct-2
"""

import json
import random
import re
from collections import Counter
from pathlib import Path

# ====================== Configuration ======================
SAMPLE_SIZE = 5000
OUTPUT_FILE = "omi_5000_classified.jsonl"
RANDOM_SEED = 1777

# ====================== Rule-based Classifier ======================
def classify_math_subject(problem: str, solution: str = "", answer: str = "") -> str:
    """Classify a math problem into subject categories using keywords."""
    text = (str(problem) + " " + str(solution) + " " + str(answer)).lower()

    # Geometry
    geometry_keywords = [
        r"\btriangle\b", r"\bcircle\b", r"\bangle\b", r"\bperimeter\b", r"\barea\b",
        r"\bvolume\b", r"\bradius\b", r"\bdiameter\b", r"\bchord\b", r"\btangent\b",
        r"\bpolygon\b", r"\bquadrilateral\b", r"\bparallelogram\b", r"\brhombus\b",
        r"\btrapezoid\b", r"\bcongruent\b", r"\bsimilar\b", r"\bpythagorean\b",
        r"\bhypotenuse\b", r"\bcoordinate\b", r"\bdistance formula\b", r"\bmidpoint\b",
        r"\bhexagon\b", r"\bpentagon\b", r"\bsector\b", r"\barc\b", r"\bcircumference\b"
    ]
    if any(re.search(k, text) for k in geometry_keywords):
        return "Geometry"

    # Number Theory
    number_theory_keywords = [
        r"\bprime\b", r"\bdivisible\b", r"\bdivisor\b", r"\bgcd\b", r"\blcm\b",
        r"\bmodulo\b", r"\bmod\b", r"\bcongruence\b", r"\bfactorial\b",
        r"\bremainder\b", r"\bodd\b", r"\beven\b", r"\bdigit\b", r"\bbase\b",
        r"\bperfect square\b", r"\bperfect cube\b", r"\bfibonacci\b",
        r"\brelatively prime\b", r"\bcoprime\b", r"\binteger solutions\b"
    ]
    if any(re.search(k, text) for k in number_theory_keywords):
        return "Number Theory"

    # Counting & Probability
    counting_keywords = [
        r"\bprobability\b", r"\bcombination\b", r"\bpermutation\b", r"\bchoose\b",
        r"\bncr\b", r"\bnpr\b", r"\bways\b", r"\barrangement\b", r"\bhow many ways\b",
        r"\bpigeonhole\b", r"\binclusion-exclusion\b", r"\bsample space\b"
    ]
    if any(re.search(k, text) for k in counting_keywords):
        return "Counting & Probability"

    # Precalculus / Calculus
    calc_keywords = [
        r"\blimit\b", r"\bderivative\b", r"\bintegral\b", r"\bdifferentiate\b",
        r"\bintegrate\b", r"\basymptote\b", r"\btrigonometric\b", r"\bsin\b", r"\bcos\b",
        r"\btan\b", r"\blogarithm\b", r"\bexponential\b", r"\bsequence\b", r"\bseries\b",
        r"\barithmetic sequence\b", r"\bgeometric sequence\b"
    ]
    if any(re.search(k, text) for k in calc_keywords):
        return "Precalculus / Calculus"

    # Intermediate Algebra
    intermediate_keywords = [
        r"\bquadratic\b", r"\bpolynomial\b", r"\bfactor\b", r"\broots\b",
        r"\bcomplex number\b", r"\bsystem of equations\b", r"\binverse function\b",
        r"\brational expression\b", r"\bpartial fraction\b", r"\bmatrix\b"
    ]
    if any(re.search(k, text) for k in intermediate_keywords):
        return "Intermediate Algebra"

    # Prealgebra
    prealgebra_keywords = [
        r"\bfraction\b", r"\bdecimal\b", r"\bpercent\b", r"\bratio\b", r"\bproportion\b",
        r"\baverage\b", r"\bmean\b", r"\bmedian\b", r"\bmode\b"
    ]
    if any(re.search(k, text) for k in prealgebra_keywords) and len(str(problem)) < 350:
        return "Prealgebra"

    # Algebra
    algebra_keywords = [
        r"\bequation\b", r"\bexpression\b", r"\bsolve for\b", r"\bvariable\b",
        r"\binquality\b", r"\blinear\b", r"\bsimplify\b", r"\bexpand\b", r"\bfunction\b"
    ]
    if any(re.search(k, text) for k in algebra_keywords):
        return "Algebra"

    return "Other / Mixed"


# ====================== Main ======================
def main():
    print("Loading OpenMathInstruct-2 (streaming)...")
    
    try:
        from datasets import load_dataset
        ds = load_dataset("nvidia/OpenMathInstruct-2", split="train", streaming=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Make sure you have run:  hf download nvidia/OpenMathInstruct-2 --repo-type dataset")
        print("And installed:  pip install datasets")
        return

    # Collect more than needed, then sample
    print(f"Collecting examples (target ~{SAMPLE_SIZE + 1000})...")
    all_examples = []
    for i, ex in enumerate(ds):
        all_examples.append({
            "problem": ex["problem"],
            "generated_solution": ex["generated_solution"],
            "expected_answer": ex["expected_answer"],
            "problem_source": ex["problem_source"],
        })
        if (i + 1) % 1000 == 0:
            print(f"  collected {i + 1}...")
        if i >= SAMPLE_SIZE + 999:
            break

    print(f"Collected {len(all_examples)} examples. Sampling {SAMPLE_SIZE}...")
    random.seed(RANDOM_SEED)
    samples = random.sample(all_examples, min(SAMPLE_SIZE, len(all_examples)))

    # Classify
    print("Classifying...")
    for ex in samples:
        ex["subject"] = classify_math_subject(
            problem=ex["problem"],
            solution=ex["generated_solution"],
            answer=ex["expected_answer"]
        )

    # ===== Analysis =====
    subjects = [s["subject"] for s in samples]
    sources = [s["problem_source"] for s in samples]

    print("\n" + "=" * 60)
    print("SUBJECT DISTRIBUTION")
    print("=" * 60)
    for subj, cnt in Counter(subjects).most_common():
        print(f"  {subj:28s} : {cnt:5d}  ({100 * cnt / len(samples):5.1f}%)")

    print("\n" + "=" * 60)
    print("PROBLEM_SOURCE DISTRIBUTION")
    print("=" * 60)
    for src, cnt in Counter(sources).most_common():
        print(f"  {src:28s} : {cnt:5d}  ({100 * cnt / len(samples):5.1f}%)")

    print("\n" + "=" * 60)
    print("SUBJECT × SOURCE (top 10)")
    print("=" * 60)
    cross = Counter((s["subject"], s["problem_source"]) for s in samples)
    for (subj, src), cnt in cross.most_common(10):
        print(f"  {subj:25s} | {src:18s} : {cnt}")

    # Save
    out_path = Path(OUTPUT_FILE)
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in samples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n✅ Saved {len(samples)} classified samples → {out_path.resolve()}")
    print("Done.")


if __name__ == "__main__":
    main()