#!/usr/bin/env python3
"""Extract FPS from detectron2 training/inference log files."""

import re
import sys
import argparse
from pathlib import Path


def extract_training_fps(log_path):
    """Extract training FPS from 'last_time: X.XXXX' fields."""
    pattern = re.compile(r'iter:\s*(\d+).*?last_time:\s*([\d.]+)')
    results = []
    with open(log_path, 'r', errors='ignore') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                iter_num = int(m.group(1))
                last_time = float(m.group(2))
                fps = 1.0 / last_time if last_time > 0 else 0
                results.append((iter_num, last_time, fps))
    return results


def extract_inference_fps(log_path):
    """Extract inference FPS from evaluator log lines.

    Returns (progress_list, final_s_per_iter, num_devices)
      progress_list: list of (done, total, s_per_iter, fps_per_device)
      final_s_per_iter: float or None
      num_devices: int
    """
    prog_pattern = re.compile(
        r'Inference done (\d+)/(\d+).*?Inference:\s*([\d.]+)\s*s/iter'
    )
    final_pattern = re.compile(
        r'Total inference pure compute time:.*?\(([\d.]+)\s*s / iter per device,\s*on\s*(\d+)\s*devices\)'
    )
    progress = []
    final_s_per_iter = None
    num_devices = 1

    with open(log_path, 'r', errors='ignore') as f:
        for line in f:
            m = prog_pattern.search(line)
            if m:
                done, total = int(m.group(1)), int(m.group(2))
                s = float(m.group(3))
                progress.append((done, total, s, 1.0 / s if s > 0 else 0))
            m2 = final_pattern.search(line)
            if m2:
                final_s_per_iter = float(m2.group(1))
                num_devices = int(m2.group(2))

    return progress, final_s_per_iter, num_devices


def scan_output_dir(root):
    """Scan an output directory for all eval sub-logs.

    Looks for:
      <root>/log.txt                    (training)
      <root>/<task>/eval-<dataset>/log.txt  (inference)

    Returns list of (label, log_path, is_inference).
    """
    root = Path(root)
    entries = []

    # Training log at root
    train_log = root / 'log.txt'
    if train_log.exists():
        entries.append(('(training)', train_log, False))

    # Inference logs: two levels deep
    for log_path in sorted(root.rglob('log.txt')):
        if log_path == train_log:
            continue
        # Label as relative path from root, drop /log.txt suffix
        label = str(log_path.relative_to(root).parent)
        entries.append((label, log_path, True))

    return entries


def print_inference_table(rows):
    """Print a summary table for all inference datasets.

    rows: list of (label, s_per_iter, num_devices)
    """
    col_w = max(len(r[0]) for r in rows) + 2
    print("\n=== Inference FPS Summary ===")
    print("%-*s  %10s  %12s  %12s" % (col_w, "Dataset", "s/iter", "FPS(1 GPU)", "FPS(total)"))
    print("-" * (col_w + 40))
    fps_list = []
    for label, s, n in rows:
        fps1 = 1.0 / s
        fps_total = n / s
        fps_list.append(fps_total)
        print("%-*s  %10.6f  %12.2f  %12.2f" % (col_w, label, s, fps1, fps_total))
    if len(rows) > 1:
        avg_s = sum(r[1] for r in rows) / len(rows)
        avg_n = rows[0][2]  # assume same device count
        avg_fps_total = avg_n / avg_s
        print("-" * (col_w + 40))
        print("%-*s  %10.6f  %12.2f  %12.2f" % (
            col_w, "Average (%d datasets)" % len(rows),
            avg_s, 1.0 / avg_s, avg_fps_total))
        print()
        print("=" * 50)
        print("  FPS (all %d GPUs, avg over %d datasets): %.2f" % (avg_n, len(rows), avg_fps_total))
        print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description='Extract FPS from detectron2 log files.\n'
                    'Pass a single log.txt, or an output directory to scan all eval logs.',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('path', nargs='?',
                        default='/gemini/space/zhaozy/libingyu/00_OVRSIS_v2/CAT-Seg-OVRSIS-PI/output_vitl_336_OVRSIS95K_baseline',
                        help='Path to log.txt or output directory')
    parser.add_argument('--mode', choices=['train', 'inference', 'all'],
                        default='all', help='Which FPS to extract (only used for single log.txt)')
    parser.add_argument('--summary', action='store_true',
                        help='Print summary statistics only (skip per-iter details)')
    args = parser.parse_args()

    p = Path(args.path)
    if not p.exists():
        print(f"Error: {p} not found", file=sys.stderr)
        sys.exit(1)

    # ── Directory mode: scan all eval logs ──────────────────────────────────
    if p.is_dir():
        entries = scan_output_dir(p)
        if not entries:
            print("No log.txt files found under", p)
            sys.exit(1)

        inference_rows = []
        for label, log_path, is_inference in entries:
            if is_inference:
                _, final_s, n_dev = extract_inference_fps(log_path)
                if final_s is not None:
                    inference_rows.append((label, final_s, n_dev))
                else:
                    print(f"[warn] no final inference time in {log_path}")
            else:
                # Training log
                train_data = extract_training_fps(log_path)
                if train_data:
                    fps_values = [r[2] for r in train_data]
                    print("=== Training FPS ===")
                    if not args.summary:
                        print("%-8s  %12s  %8s" % ("iter", "last_time(s)", "FPS"))
                        print("-" * 34)
                        for iter_num, t, fps in train_data:
                            print("%-8d  %12.4f  %8.2f" % (iter_num, t, fps))
                    print("Summary: min=%.2f  max=%.2f  avg=%.2f  final=%.2f  samples=%d" % (
                        min(fps_values), max(fps_values),
                        sum(fps_values) / len(fps_values),
                        fps_values[-1], len(fps_values)))

        if inference_rows:
            print_inference_table(inference_rows)
        return

    # ── Single file mode ────────────────────────────────────────────────────
    log_path = p
    if args.mode in ('train', 'all'):
        train_data = extract_training_fps(log_path)
        if train_data:
            fps_values = [r[2] for r in train_data]
            print("=== Training FPS (1 / last_time) ===")
            if not args.summary:
                print("%-8s  %12s  %8s" % ("iter", "last_time(s)", "FPS"))
                print("-" * 34)
                for iter_num, t, fps in train_data:
                    print("%-8d  %12.4f  %8.2f" % (iter_num, t, fps))
            print("\nSummary: min=%.2f  max=%.2f  avg=%.2f  final=%.2f  samples=%d" % (
                min(fps_values), max(fps_values),
                sum(fps_values) / len(fps_values),
                fps_values[-1], len(fps_values)))
        else:
            print("No training FPS data found.")

    if args.mode in ('inference', 'all'):
        progress, final_s, n_dev = extract_inference_fps(log_path)
        print("\n=== Inference FPS ===")
        if progress and not args.summary:
            print("%-6s  %-6s  %11s  %8s" % ("done", "total", "inf_time(s)", "FPS"))
            print("-" * 38)
            for done, total, t, fps in progress:
                print("%-6d  %-6d  %11.4f  %8.2f" % (done, total, t, fps))
        if final_s is not None:
            print("\nFinal pure-compute FPS — per device: %.2f  total (%d GPUs): %.2f" % (
                1.0 / final_s, n_dev, n_dev / final_s))
        elif not progress:
            print("No inference FPS data found.")


if __name__ == '__main__':
    main()
