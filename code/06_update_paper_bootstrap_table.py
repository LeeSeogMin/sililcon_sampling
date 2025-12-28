"""
Update docs/journal_paper.md Section 4.7 bootstrap JS table from saved results.

Reads CLOVA response sets from:
  results/clova_experiment/<VARIABLE>/clova_results.json

Computes:
  - JS point estimate (same definition as ss_utils.js_divergence; natural log)
  - Bootstrap percentile 95% CI (2.5%, 97.5%)
  - Bootstrap SD (std of bootstrap statistics)

Then replaces the Markdown content between:
  <!-- AUTO:BOOTSTRAP_JS_TABLE_START -->
  <!-- AUTO:BOOTSTRAP_JS_TABLE_END -->
in the paper.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ss_utils import Benchmark, js_divergence, load_benchmark  # noqa: E402


@dataclass(frozen=True)
class BootstrapRow:
    variable: str
    js_point: float
    ci_low: float
    ci_high: float
    boot_sd: float


def js_from_responses(
    benchmark: Benchmark,
    variable: str,
    responses: Sequence[int],
) -> float:
    categories = benchmark.categories(variable)
    q = benchmark.distribution_prob(variable)
    sample = np.asarray(responses, dtype=int)
    counts = np.asarray([(sample == c).sum() for c in categories], dtype=float)
    p = counts / float(counts.sum())
    return js_divergence(p, q)


def bootstrap_js(
    benchmark: Benchmark,
    variable: str,
    responses: Sequence[int],
    *,
    n_boot: int,
    seed: int,
) -> BootstrapRow:
    rng = np.random.default_rng(seed)
    sample = np.asarray(responses, dtype=int)
    n = int(sample.size)
    if n <= 0:
        raise ValueError(f"Empty responses for {variable}")

    point = float(js_from_responses(benchmark, variable, sample))

    stats = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        boot = rng.choice(sample, size=n, replace=True)
        stats[i] = js_from_responses(benchmark, variable, boot)

    ci_low, ci_high = np.quantile(stats, [0.025, 0.975])
    boot_sd = float(stats.std(ddof=1))
    return BootstrapRow(
        variable=variable,
        js_point=point,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        boot_sd=boot_sd,
    )


def load_clova_responses(path: str) -> list[int]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", [])
    if not results:
        raise ValueError(f"Missing results in {path}")
    responses = results[0].get("responses", [])
    if not isinstance(responses, list) or not responses:
        raise ValueError(f"Missing responses in {path}")
    return [int(x) for x in responses]


def format_row(row: BootstrapRow, *, decimals: int) -> str:
    fmt = f"{{:.{decimals}f}}"
    point = fmt.format(row.js_point)
    ci_low = fmt.format(row.ci_low)
    ci_high = fmt.format(row.ci_high)
    sd = fmt.format(row.boot_sd)
    return f"| {row.variable} | {point} | [{ci_low}, {ci_high}] | {sd} |"


def build_table(rows: Iterable[BootstrapRow], *, decimals: int) -> str:
    header = "| Variable | JS Point | 95% CI | Bootstrap SD |"
    sep = "|----------|---------|--------|-----|"
    lines = [header, sep]
    for row in rows:
        lines.append(format_row(row, decimals=decimals))
    return "\n".join(lines) + "\n"


def replace_between_markers(text: str, start_marker: str, end_marker: str, replacement: str) -> str:
    start = text.find(start_marker)
    if start < 0:
        raise ValueError(f"Start marker not found: {start_marker}")
    end = text.find(end_marker, start)
    if end < 0:
        raise ValueError(f"End marker not found: {end_marker}")

    start_content = start + len(start_marker)
    return text[:start_content] + "\n" + replacement + text[end:]


def main() -> int:
    parser = argparse.ArgumentParser(description="Update Section 4.7 bootstrap JS table in the paper.")
    parser.add_argument("--paper", default="docs/journal_paper.md", help="Path to the paper Markdown file")
    parser.add_argument("--benchmark", default="data/kgss_benchmarks_2023.json", help="Path to benchmark JSON")
    parser.add_argument("--results-root", default="results/clova_experiment", help="Base directory of CLOVA results")
    parser.add_argument(
        "--variables",
        nargs="+",
        default=["CONFINAN", "CONLEGIS", "KRPROUD", "NORTHWHO", "UNIFI", "PARTYLR"],
        help="Variables to include in the table (order preserved)",
    )
    parser.add_argument("--n-boot", type=int, default=5000, help="Number of bootstrap resamples")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for bootstrap resampling")
    parser.add_argument("--decimals", type=int, default=3, help="Decimal places for table values")
    parser.add_argument("--write", action="store_true", help="Write changes to the paper file")
    args = parser.parse_args()

    benchmark = load_benchmark(args.benchmark)

    rows: list[BootstrapRow] = []
    for variable in args.variables:
        path = os.path.join(args.results_root, variable, "clova_results.json")
        responses = load_clova_responses(path)
        rows.append(bootstrap_js(benchmark, variable, responses, n_boot=args.n_boot, seed=args.seed))

    table = build_table(rows, decimals=args.decimals)

    with open(args.paper, "r", encoding="utf-8") as f:
        paper = f.read()

    start_marker = "<!-- AUTO:BOOTSTRAP_JS_TABLE_START -->"
    end_marker = "<!-- AUTO:BOOTSTRAP_JS_TABLE_END -->"
    updated = replace_between_markers(paper, start_marker, end_marker, table)

    if not args.write:
        print(table, end="")
        return 0

    with open(args.paper, "w", encoding="utf-8") as f:
        f.write(updated)

    print(f"Updated {args.paper} ({len(rows)} variables, n_boot={args.n_boot}, seed={args.seed}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
