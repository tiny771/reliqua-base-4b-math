#!/usr/bin/env python3
"""
Calibration analysis script for perplexity and difficulty thresholds.

Analyzes historical mining logs to determine optimal gate thresholds that maximize zone-pass rate.

Usage:
    python analyze_success.py [--log-file path/to/miner_analysis.jsonl] [--output report.txt]
"""

import json
import sys
import statistics
from pathlib import Path
from collections import defaultdict
from argparse import ArgumentParser


def analyze_logs(log_file_path):
    """Parse JSONL log file and categorize records by success status."""
    submitted = []
    failed = []
    preflight_failed = []
    
    try:
        with open(log_file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    rec = json.loads(line.strip())
                    if not rec:
                        continue
                    
                    status = rec.get("status", "")
                    
                    if status == "submitted":
                        submitted.append(rec)
                    elif status in ["sigma_skipped", "rollout_incomplete", "submit_failed", "preflight_skipped"]:
                        failed.append(rec)
                    elif status == "preflight_failed":
                        preflight_failed.append(rec)
                        
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON at line {line_num}: {e}", file=sys.stderr)
                    
    except FileNotFoundError:
        print(f"Error: Log file not found: {log_file_path}", file=sys.stderr)
        return None, None, None
    
    return submitted, failed, preflight_failed


def calculate_stats(records, field_name):
    """Calculate statistics for a field across records."""
    values = [r.get(field_name) for r in records if field_name in r and r[field_name] is not None]
    
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "stdev": None,
        }
    
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0,
    }


def find_perplexity_threshold(submitted, failed, threshold_range=None):
    """Find optimal perplexity threshold to maximize zone-pass rate."""
    if threshold_range is None:
        # Auto-detect range from data
        all_perp = [r.get("perplexity") for r in submitted + failed 
                    if r.get("perplexity") is not None]
        if not all_perp:
            return None, []
        
        min_perp = min(all_perp)
        max_perp = max(all_perp)
        threshold_range = [min_perp, min_perp + (max_perp - min_perp) * 0.3, 
                          min_perp + (max_perp - min_perp) * 0.5,
                          min_perp + (max_perp - min_perp) * 0.7, max_perp]
    
    results = []
    for threshold in threshold_range:
        # Count passes and fails below/above threshold
        passes_below = sum(1 for r in submitted if r.get("perplexity", float('inf')) < threshold)
        fails_below = sum(1 for r in failed if r.get("perplexity", float('inf')) < threshold)
        
        total_below = passes_below + fails_below
        if total_below > 0:
            pass_rate = passes_below / total_below
        else:
            pass_rate = 0.0
        
        passes_total = len(submitted)
        coverage = passes_below / passes_total if passes_total > 0 else 0
        
        results.append({
            "threshold": threshold,
            "zone_passes": passes_below,
            "zone_fails": fails_below,
            "pass_rate": pass_rate,
            "coverage": coverage,  # % of submitted prompts captured
        })
    
    return results


def find_difficulty_range(submitted, failed, step_size=0.5):
    """Find optimal difficulty range to maximize zone-pass rate."""
    all_diff = [r.get("diff") for r in submitted + failed 
                if r.get("diff") is not None]
    
    if not all_diff:
        return []
    
    min_diff = min(all_diff)
    max_diff = max(all_diff)
    
    results = []
    current_lo = min_diff
    
    while current_lo < max_diff:
        current_hi = min(current_lo + step_size, max_diff + step_size)
        
        passes = sum(1 for r in submitted 
                    if current_lo <= r.get("diff", -1) < current_hi)
        fails = sum(1 for r in failed 
                   if current_lo <= r.get("diff", -1) < current_hi)
        
        total = passes + fails
        if total > 0:
            pass_rate = passes / total
        else:
            pass_rate = 0.0
        
        passes_total = len(submitted)
        coverage = passes / passes_total if passes_total > 0 else 0
        
        results.append({
            "range": (current_lo, current_hi),
            "zone_passes": passes,
            "zone_fails": fails,
            "pass_rate": pass_rate,
            "coverage": coverage,
        })
        
        current_lo += step_size
    
    return results


def generate_report(submitted, failed, preflight_failed, output_file=None):
    """Generate comprehensive calibration report."""
    report_lines = []
    
    report_lines.append("=" * 80)
    report_lines.append("RELIQUARY MINER CALIBRATION ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Summary statistics
    report_lines.append("📊 SUMMARY STATISTICS")
    report_lines.append("-" * 80)
    report_lines.append(f"Total records analyzed: {len(submitted) + len(failed) + len(preflight_failed)}")
    report_lines.append(f"  • Submitted (zone-pass): {len(submitted)}")
    report_lines.append(f"  • Failed (zone-fail): {len(failed)}")
    report_lines.append(f"  • Preflight failed: {len(preflight_failed)}")
    
    if len(submitted) + len(failed) > 0:
        overall_pass_rate = len(submitted) / (len(submitted) + len(failed))
        report_lines.append(f"  • Overall zone-pass rate: {overall_pass_rate:.1%}")
    report_lines.append("")
    
    # Perplexity analysis
    report_lines.append("📈 PERPLEXITY ANALYSIS")
    report_lines.append("-" * 80)
    
    perp_submitted = calculate_stats(submitted, "perplexity")
    perp_failed = calculate_stats(failed, "perplexity")
    
    report_lines.append("Submitted (zone-pass) perplexity:")
    if perp_submitted["count"] > 0:
        report_lines.append(f"  • Count: {perp_submitted['count']}")
        report_lines.append(f"  • Mean: {perp_submitted['mean']:.3f}")
        report_lines.append(f"  • Median: {perp_submitted['median']:.3f}")
        report_lines.append(f"  • Range: [{perp_submitted['min']:.3f}, {perp_submitted['max']:.3f}]")
        report_lines.append(f"  • Stdev: {perp_submitted['stdev']:.3f}")
    else:
        report_lines.append("  • No data")
    report_lines.append("")
    
    report_lines.append("Failed (zone-fail) perplexity:")
    if perp_failed["count"] > 0:
        report_lines.append(f"  • Count: {perp_failed['count']}")
        report_lines.append(f"  • Mean: {perp_failed['mean']:.3f}")
        report_lines.append(f"  • Median: {perp_failed['median']:.3f}")
        report_lines.append(f"  • Range: [{perp_failed['min']:.3f}, {perp_failed['max']:.3f}]")
        report_lines.append(f"  • Stdev: {perp_failed['stdev']:.3f}")
    else:
        report_lines.append("  • No data")
    report_lines.append("")
    
    # Perplexity threshold recommendations
    perp_results = find_perplexity_threshold(submitted, failed)
    if perp_results:
        report_lines.append("Perplexity threshold analysis:")
        report_lines.append(f"{'Threshold':<12} {'Passes':<8} {'Fails':<8} {'Pass Rate':<12} {'Coverage':<10}")
        report_lines.append("-" * 60)
        
        best_result = None
        best_score = -1
        
        for res in perp_results:
            report_lines.append(f"{res['threshold']:<12.3f} {res['zone_passes']:<8} {res['zone_fails']:<8} "
                              f"{res['pass_rate']:<12.1%} {res['coverage']:<10.1%}")
            
            # Score = balance between pass rate and coverage
            score = (res['pass_rate'] * 0.6) + (res['coverage'] * 0.4)
            if score > best_score:
                best_score = score
                best_result = res
        
        if best_result:
            report_lines.append("")
            report_lines.append(f"✓ RECOMMENDED perplexity threshold: {best_result['threshold']:.3f}")
            report_lines.append(f"  Expected zone-pass rate: {best_result['pass_rate']:.1%}")
            report_lines.append(f"  Will capture: {best_result['coverage']:.1%} of successful prompts")
    report_lines.append("")
    
    # Difficulty analysis
    report_lines.append("📊 DIFFICULTY ANALYSIS")
    report_lines.append("-" * 80)
    
    diff_submitted = calculate_stats(submitted, "diff")
    diff_failed = calculate_stats(failed, "diff")
    
    report_lines.append("Submitted (zone-pass) difficulty:")
    if diff_submitted["count"] > 0:
        report_lines.append(f"  • Count: {diff_submitted['count']}")
        report_lines.append(f"  • Mean: {diff_submitted['mean']:.3f}")
        report_lines.append(f"  • Median: {diff_submitted['median']:.3f}")
        report_lines.append(f"  • Range: [{diff_submitted['min']:.3f}, {diff_submitted['max']:.3f}]")
        report_lines.append(f"  • Stdev: {diff_submitted['stdev']:.3f}")
    else:
        report_lines.append("  • No data")
    report_lines.append("")
    
    report_lines.append("Failed (zone-fail) difficulty:")
    if diff_failed["count"] > 0:
        report_lines.append(f"  • Count: {diff_failed['count']}")
        report_lines.append(f"  • Mean: {diff_failed['mean']:.3f}")
        report_lines.append(f"  • Median: {diff_failed['median']:.3f}")
        report_lines.append(f"  • Range: [{diff_failed['min']:.3f}, {diff_failed['max']:.3f}]")
        report_lines.append(f"  • Stdev: {diff_failed['stdev']:.3f}")
    else:
        report_lines.append("  • No data")
    report_lines.append("")
    
    # Difficulty range recommendations
    diff_results = find_difficulty_range(submitted, failed)
    if diff_results:
        report_lines.append("Difficulty range analysis (top 10 best ranges):")
        report_lines.append(f"{'Range (Lo, Hi)':<20} {'Passes':<8} {'Fails':<8} {'Pass Rate':<12} {'Coverage':<10}")
        report_lines.append("-" * 70)
        
        # Sort by pass rate * coverage
        sorted_results = sorted(diff_results, 
                               key=lambda r: (r['pass_rate'] * 0.6) + (r['coverage'] * 0.4),
                               reverse=True)
        
        best_result = None
        for i, res in enumerate(sorted_results[:10]):
            lo, hi = res['range']
            report_lines.append(f"[{lo:<6.2f}, {hi:<6.2f}) {res['zone_passes']:<8} {res['zone_fails']:<8} "
                              f"{res['pass_rate']:<12.1%} {res['coverage']:<10.1%}")
            if i == 0:
                best_result = res
        
        if best_result:
            lo, hi = best_result['range']
            report_lines.append("")
            report_lines.append(f"✓ RECOMMENDED difficulty range: [{lo:.2f}, {hi:.2f})")
            report_lines.append(f"  Expected zone-pass rate: {best_result['pass_rate']:.1%}")
            report_lines.append(f"  Will capture: {best_result['coverage']:.1%} of successful prompts")
    report_lines.append("")
    
    # Implementation recommendations
    report_lines.append("🔧 IMPLEMENTATION RECOMMENDATIONS")
    report_lines.append("-" * 80)
    report_lines.append("")
    report_lines.append("1. Update engine.py line 356:")
    if perp_results and perp_results[0]:
        threshold = perp_results[0]['threshold']
        report_lines.append(f"   self._preflight_perplexity_threshold: float = {threshold:.3f}")
    report_lines.append("")
    report_lines.append("2. Set difficulty range in CLI or config:")
    if diff_results and diff_results[0]:
        lo, hi = diff_results[0]['range']
        report_lines.append(f"   --difficulty-range {lo:.2f},{hi:.2f}")
    report_lines.append("")
    report_lines.append("3. Run A/B test:")
    report_lines.append("   • 5 windows with gates OFF (baseline)")
    report_lines.append("   • 5 windows with gates ON (with calibrated thresholds)")
    report_lines.append("   • Compare zone-pass rate and emission")
    report_lines.append("")
    
    report_lines.append("=" * 80)
    
    report_text = "\n".join(report_lines)
    
    if output_file:
        try:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"✓ Report written to {output_file}")
        except Exception as e:
            print(f"Error writing to {output_file}: {e}", file=sys.stderr)
    
    print(report_text)
    return report_text


def main():
    parser = ArgumentParser(description="Calibration analysis for perplexity and difficulty gates")
    parser.add_argument("--log-file", type=str, default="miner_analysis.jsonl",
                       help="Path to JSONL log file (default: miner_analysis.jsonl)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file for report (default: stdout only)")
    
    args = parser.parse_args()
    
    log_path = Path(args.log_file)
    if not log_path.exists():
        # Try relative to parent directories
        for parent in Path.cwd().parents:
            candidate = parent / "results" / "miner_analysis.jsonl"
            if candidate.exists():
                log_path = candidate
                break
    
    print(f"Loading logs from: {log_path}")
    submitted, failed, preflight_failed = analyze_logs(str(log_path))
    
    if submitted is None:
        sys.exit(1)
    
    print(f"Loaded: {len(submitted)} submitted, {len(failed)} failed, {len(preflight_failed)} preflight_failed")
    print("")
    
    generate_report(submitted, failed, preflight_failed, args.output)


if __name__ == "__main__":
    main()
