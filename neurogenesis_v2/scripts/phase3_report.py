#!/usr/bin/env python3
"""Combined Phase 3 report — runs eval + generation + documents findings."""
import os
import sys
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def main():
    os.makedirs('logs', exist_ok=True)
    scripts_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 70)
    print("NEUROGENESIS V2 — PHASE 3 SUMMARY REPORT")
    print("=" * 70)

    # Run eval
    print("\n[1/2] Running Phase 3 evaluation...")
    subprocess.run([sys.executable, os.path.join(scripts_dir, 'eval_phase3.py')], check=True)

    # Run generation comparison
    print("\n[2/2] Running generation comparison...")
    subprocess.run([sys.executable, os.path.join(scripts_dir, 'generation_comparison.py')], check=True)

    # Combine reports
    report_parts = []
    for fname in ['logs/phase3_eval_report.txt', 'logs/generation_comparison.txt']:
        if os.path.exists(fname):
            with open(fname) as f:
                report_parts.append(f.read())

    combined = '\n\n'.join(report_parts)
    with open('logs/phase3_report.txt', 'w') as f:
        f.write(combined)

    print(f"\n{'='*70}")
    print(f"Combined report saved to logs/phase3_report.txt")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
