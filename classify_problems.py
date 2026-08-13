import json
import os
from pathlib import Path
from typing import Optional
import re

SUBJECTS = [
    "Algebra",
    "Geometry",
    "Number Theory",
    "Counting & Probability",
    "Intermediate Algebra",
    "Prealgebra",
    "Precalculus / Calculus",
    "Other /Mixed"
]

INPUT_FILE = "cooldown_problems.jsonl"
OUTPUT_FILE = "cooldown_problems_classified.jsonl"

def classify_math_subject_rule_based(
    problem: str,
    generated_solution: str,
    expected_answer: str=""
) -> str:
    """
    Classify a math problem into subjects using keyords heuristics.
    Uses problem, solution and answer for better accuracy.
    """
    text = (problem + " " + generated_solution + " " + expected_answer).lower()

    # ---------- Strong Geometry signals ----------
    geometry_keywords = [
        r"\btriangle\b", r"\bcircle\b", r"\bangle\b", r"\bperimeter\b", r"\barea\b",
        r"\bvolume\b", r"\bradius\b", r"\bdiameter\b", r"\bchord\b", r"\btangent\b",
        r"\bsecant\b", r"\bpolygon\b", r"\bquadrilateral\b", r"\bparallelogram\b",
        r"\brhombus\b", r"\btrapezoid\b", r"\bhexagon\b", r"\bpentagon\b",
        r"\bcongruent\b", r"\bsimilar\b", r"\bpythagorean\b", r"\bhypotenuse\b",
        r"\bcoordinate geometry\b", r"\bdistance formula\b", r"\bmidpoint\b",
        r"\bcircumscribed\b", r"\binscribed\b", r"\bsector\b", r"\barc\b"
    ]
    if any(re.search(k, text) for k in geometry_keywords):
        return "Geometry"

    # ---------- Number Theory ----------
    number_theory_keywords = [
        r"\bprime\b", r"\bdivisible\b", r"\bdivisor\b", r"\bgcd\b", r"\blcm\b",
        r"\bmodulo\b", r"\bmod\b", r"\bcongruence\b", r"\bfactorial\b",
        r"\bremainder\b", r"\bodd\b", r"\beven\b", r"\bdigit\b", r"\bbase\b",
        r"\bperfect square\b", r"\bperfect cube\b", r"\bfibonacci\b",
        r"\bnumber of positive divisors\b", r"\brelatively prime\b",
        r"\bcoprime\b", r"\binteger solutions\b"
    ]
    if any(re.search(k, text) for k in number_theory_keywords):
        return "Number Theory"

    # ---------- Counting & Probability ----------
    counting_keywords = [
        r"\bprobability\b", r"\bcombination\b", r"\bpermutation\b", r"\bchoose\b",
        r"\bncr\b", r"\bnpr\b", r"\bfactorial\b", r"\bways\b", r"\barrangement\b",
        r"\bhow many ways\b", r"\bpigeonhole\b", r"\binclusion-exclusion\b",
        r"\bsample space\b", r"\bevent\b", r"\bindependent\b", r"\bmutually exclusive\b"
    ]
    if any(re.search(k, text) for k in counting_keywords):
        return "Counting & Probability"

    # ---------- Precalculus / Calculus ----------
    calc_keywords = [
        r"\blimit\b", r"\bderivative\b", r"\bintegral\b", r"\bdifferentiate\b",
        r"\bintegrate\b", r"\bcontinuity\b", r"\basymptote\b", r"\btrigonometric\b",
        r"\bsin\b", r"\bcos\b", r"\btan\b", r"\bsec\b", r"\bcsc\b", r"\bcot\b",
        r"\blogarithm\b", r"\bexponential\b", r"\bsequence\b", r"\bseries\b",
        r"\barithmetic sequence\b", r"\bgeometric sequence\b", r"\bsummation\b"
    ]
    if any(re.search(k, text) for k in calc_keywords):
        return "Precalculus / Calculus"

    # ---------- Intermediate Algebra (more advanced algebra) ----------
    intermediate_alg_keywords = [
        r"\bquadratic\b", r"\bpolynomial\b", r"\bfactor\b", r"\broots\b",
        r"\bcomplex number\b", r"\bi\b", r"\bmatrix\b", r"\bdeterminant\b",
        r"\bsystem of equations\b", r"\binverse function\b", r"\bcomposition\b",
        r"\blogarithmic equation\b", r"\bexponential equation\b",
        r"\brational expression\b", r"\bpartial fraction\b"
    ]
    if any(re.search(k, text) for k in intermediate_alg_keywords):
        return "Intermediate Algebra"

    # ---------- Prealgebra (basic) ----------
    prealgebra_keywords = [
        r"\bfraction\b", r"\bdecimal\b", r"\bpercent\b", r"\bratio\b", r"\bproportion\b",
        r"\baverage\b", r"\bmean\b", r"\bmedian\b", r"\bmode\b", r"\border of operations\b",
        r"\binteger\b", r"\bwhole number\b", r"\bpositive integer\b"
    ]
    # Only classify as Prealgebra if it looks simple and no stronger signal
    if any(re.search(k, text) for k in prealgebra_keywords) and len(problem) < 300:
        return "Prealgebra"

    # ---------- Algebra (default for equations, expressions, etc.) ----------
    algebra_keywords = [
        r"\bequation\b", r"\bexpression\b", r"\bsolve for\b", r"\bvariable\b",
        r"\bx\b", r"\by\b", r"\binquality\b", r"\blinear\b", r"\bsimplify\b",
        r"\bexpand\b", r"\bfactor\b", r"\bfunction\b"
    ]
    if any(re.search(k, text) for k in algebra_keywords):
        return "Algebra"

    # Fallback
    return "Other / Mixed"

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"File not found: {INPUT_FILE}")
        return

    classified = []
    stats = {}

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line.strip())

                problem_text = item.get("problem", "")
                solution_text = item.get("solution", "")
                answer_text = item.get("ground_truth", "")

                subject = classify_math_subject_rule_based(
                    problem=problem_text,
                    generated_solution=solution_text,
                    expected_answer=answer_text
                )

                item["subject"] = subject
                classified.append(item)

                stats[subject] = stats.get(subject, 0) + 1

            except Exception as e:
                print(f"Error on line {line_num}: {e}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in classified:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(classified)} classified problems -> {OUTPUT_FILE}")
    print("\nSubject distribution:")
    for subject, count in sorted(stats.items(), key=lambda x: -x[1]):
        print(f" {subject}: {count}")

if __name__ == "__main__":
    main()