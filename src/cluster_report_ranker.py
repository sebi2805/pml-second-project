from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReportSummary:
    path: Path
    clusters: int
    samples: int
    samples_total: int
    purity_weighted: float
    entropy_weighted: float
    purity_macro: float
    entropy_macro: float
    purity_coverage: float
    coverage: float
    noise_ratio: float
    noise_dominant: bool


def load_summary(path: Path, include_noise: bool) -> ReportSummary | None:
    sizes: list[int] = []
    purities: list[float] = []
    entropies: list[float] = []

    total_samples = 0
    noise_samples = 0
    noise_size = 0
    max_non_noise_size = 0

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            cluster = (row.get("cluster") or "").strip()
            if not cluster:
                continue
            try:
                size = int(float(row.get("size", 0) or 0))
                purity = float(row.get("purity", 0) or 0)
                entropy = float(row.get("entropy", 0) or 0)
            except ValueError:
                continue
            if size <= 0:
                continue

            total_samples += size
            if cluster == "-1":
                noise_samples += size
                noise_size = size
                if not include_noise:
                    continue
            else:
                if size > max_non_noise_size:
                    max_non_noise_size = size

            sizes.append(size)
            purities.append(purity)
            entropies.append(entropy)

    if total_samples == 0:
        return None

    if sizes:
        total_used = sum(sizes)
        purity_weighted = sum(p * s for p, s in zip(purities, sizes, strict=False)) / total_used
        entropy_weighted = sum(e * s for e, s in zip(entropies, sizes, strict=False)) / total_used
        purity_macro = sum(purities) / len(purities)
        entropy_macro = sum(entropies) / len(entropies)
    else:
        total_used = 0
        purity_weighted = 0.0
        entropy_weighted = 0.0
        purity_macro = 0.0
        entropy_macro = 0.0

    coverage = (total_samples - noise_samples) / total_samples if total_samples else 0.0
    noise_ratio = noise_samples / total_samples if total_samples else 0.0
    purity_coverage = purity_weighted * coverage
    noise_dominant = noise_size >= max_non_noise_size and noise_size > 0

    return ReportSummary(
        path=path,
        clusters=len(sizes),
        samples=total_used,
        samples_total=total_samples,
        purity_weighted=purity_weighted,
        entropy_weighted=entropy_weighted,
        purity_macro=purity_macro,
        entropy_macro=entropy_macro,
        purity_coverage=purity_coverage,
        coverage=coverage,
        noise_ratio=noise_ratio,
        noise_dominant=noise_dominant,
    )


def collect_summaries(root: Path, pattern: str, include_noise: bool) -> list[ReportSummary]:
    summaries: list[ReportSummary] = []
    for path in root.rglob(pattern):
        if not path.is_file():
            continue
        summary = load_summary(path, include_noise)
        if summary is not None:
            summaries.append(summary)
    return summaries


def filter_by_coverage(summaries: list[ReportSummary], min_coverage: float) -> list[ReportSummary]:
    if min_coverage <= 0:
        return summaries
    return [summary for summary in summaries if summary.coverage >= min_coverage]

def filter_noise_dominant(summaries: list[ReportSummary], allow_noise_dominant: bool) -> list[ReportSummary]:
    if allow_noise_dominant:
        return summaries
    return [summary for summary in summaries if not summary.noise_dominant]


def print_ranked(title: str, summaries: list[ReportSummary], key_name: str, reverse: bool, top: int) -> None:
    print(f"\n{title}")
    sorted_items = sorted(summaries, key=lambda s: getattr(s, key_name), reverse=reverse)
    for idx, summary in enumerate(sorted_items[:top], start=1):
        print(
            f"{idx}. {summary.path} "
            f"purity={summary.purity_weighted:.4f} "
            f"entropy={summary.entropy_weighted:.4f} "
            f"purity*cov={summary.purity_coverage:.4f} "
            f"coverage={summary.coverage:.3f} noise={summary.noise_ratio:.3f} "
            f"clusters={summary.clusters} samples={summary.samples}/{summary.samples_total}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank cluster report CSVs by purity and entropy with noise penalty."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("output"),
        help="Root directory to search for *_clusters.csv files.",
    )
    parser.add_argument(
        "--pattern",
        default="*_clusters.csv",
        help="Glob pattern to match report files.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of best results to show per metric.",
    )
    parser.add_argument(
        "--include-noise",
        action="store_true",
        help="Include cluster -1 rows (DBSCAN noise) in purity/entropy averages.",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.0,
        help="Minimum non-noise coverage required for purity/entropy rankings.",
    )
    parser.add_argument(
        "--allow-noise-dominant",
        action="store_true",
        help="Keep reports where noise is the largest cluster.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root
    summaries = collect_summaries(root, args.pattern, args.include_noise)
    if not summaries:
        print(f"No cluster reports found under {root} with pattern {args.pattern}.")
        return

    summaries = filter_noise_dominant(summaries, args.allow_noise_dominant)
    if not summaries:
        print("All reports were noise-dominant; nothing to rank.")
        return

    print(f"Found {len(summaries)} cluster reports under {root}.")
    print_ranked(
        "Best by purity * coverage (higher is better)",
        summaries,
        key_name="purity_coverage",
        reverse=True,
        top=args.top,
    )

    filtered = filter_by_coverage(summaries, args.min_coverage)
    if args.min_coverage > 0 and not filtered:
        print(f"No reports meet min coverage >= {args.min_coverage:.3f}.")
        return

    print_ranked(
        "Best by weighted purity (higher is better)",
        filtered,
        key_name="purity_weighted",
        reverse=True,
        top=args.top,
    )
    print_ranked(
        "Best by weighted entropy (lower is better)",
        filtered,
        key_name="entropy_weighted",
        reverse=False,
        top=args.top,
    )


if __name__ == "__main__":
    main()
