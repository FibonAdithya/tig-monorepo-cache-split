#!/usr/bin/env python3
"""Fit the vector_search quality constants so GAN instances land in today's band.

    quality = (QUALITY_OFFSET - avg_dist) / QUALITY_SCALE

Naively matching the optimal-vs-random spread is the wrong target. Live mainnet
data shows competing algorithms sit within ~80 quality units of each other out
of ~72,000, all within ~1% of exact 1-NN, while quality drifts ~5,800 units
across the active track range. What must be preserved is therefore the
achievable level at each track, which is what this fits.

Two unknowns against five tracks, so it is a least-squares fit; the residuals
say whether fixed constants suffice or whether they must vary with n_queries.

Input is the per-nonce JSONL produced by measuring real generated instances
(one object per line, `label` of the form "<n_queries>:<nonce>", plus `optimal`,
`zero` and `random` average distances). Per-nonce rather than one aggregate per
track on purpose: the nonce-to-nonce spread, converted to quality units, is the
number that says whether the fitted scale leaves a playable competition, and it
is invisible in a pre-averaged input.

Deliberately stdlib-only. The fit is a two-parameter regression, and a numpy
dependency would confine it to the GPU box for no benefit.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

QUALITY_PRECISION = 1_000_000

# Median qualifier quality per track on mainnet, block 1298164 (44-53 qualifiers
# each). These are what GAN instances must reproduce. Medians rather than maxima:
# the target is where the field sits, not the single best nonce.
#
#   track   n   min     max     median
#    7000  44   71840   71920   71862
#    9000  42   73718   73801   73739
#   11000  50   75210   75291   75234
#   13000  45   76501   76600   76523
#   15000  53   77664   77778   77696
#
# Do not interpolate these. The observed spread within a track is only ~80-110
# units, so a guessed value is wrong by more than the entire competitive range.
MAINNET_MEDIAN = {
    7000: 71_862,
    9000: 73_739,
    11000: 75_234,
    13000: 76_523,
    15000: 77_696,
}

MAINNET_WITHIN_TRACK_SPREAD = {
    7000: 71_920 - 71_840,
    9000: 73_801 - 73_718,
    11000: 75_291 - 75_210,
    13000: 76_600 - 76_501,
    15000: 77_778 - 77_664,
}

MIN_ACTIVE_QUALITY = 68_500


def load(path):
    """Group per-nonce measurements by track."""
    by_track = defaultdict(list)
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        track = int(row["label"].split(":")[0])
        by_track[track].append(row)
    return by_track


def mean(values):
    return sum(values) / len(values)


def fit(points):
    """Least-squares fit of quality = a + b * (-avg_dist).

    Returns (offset, scale). Substituting q = A*offset - A*avg gives b = A and
    a = A*offset, so scale = 1/b and offset = a/b.
    """
    xs = [-avg for avg, _ in points]
    ys = [q for _, q in points]
    x_bar, y_bar = mean(xs), mean(ys)
    sxx = sum((x - x_bar) ** 2 for x in xs)
    if sxx == 0.0:
        raise SystemExit("all tracks have the same avg_dist; cannot fit a slope")
    sxy = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = y_bar - b * x_bar
    return a / b, 1.0 / b


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "measurements",
        type=Path,
        help='per-nonce JSONL: {"label": "7000:0", "optimal": 1.15, ...}',
    )
    args = p.parse_args()

    by_track = load(args.measurements)
    missing = set(MAINNET_MEDIAN) - set(by_track)
    if missing:
        raise SystemExit(f"no measurements for tracks {sorted(missing)}")

    points = [
        (mean([r["optimal"] for r in by_track[t]]), MAINNET_MEDIAN[t] / QUALITY_PRECISION)
        for t in sorted(MAINNET_MEDIAN)
    ]
    offset, scale = fit(points)

    def quality(avg_dist):
        return round((offset - avg_dist) / scale * QUALITY_PRECISION)

    print(f"QUALITY_OFFSET = {offset:.6f}")
    print(f"QUALITY_SCALE  = {scale:.6f}")
    print()
    print("Fit against the mainnet medians:")
    print(f"{'track':>7} {'n':>3} {'avg_opt':>10} {'target':>8} {'fitted':>8} {'resid':>7}")
    worst = 0
    for track in sorted(MAINNET_MEDIAN):
        rows = by_track[track]
        avg = mean([r["optimal"] for r in rows])
        fitted = quality(avg)
        resid = fitted - MAINNET_MEDIAN[track]
        worst = max(worst, abs(resid))
        print(
            f"{track:>7} {len(rows):>3} {avg:>10.6f} "
            f"{MAINNET_MEDIAN[track]:>8} {fitted:>8} {resid:>7}"
        )
    print(f"\nworst residual: {worst} quality units")
    print("Residuals materially above ~100 mean fixed constants do not suffice")
    print("and the constants must vary with n_queries.")

    print("\nBaseline solutions -- all must fall below min_active_quality "
          f"({MIN_ACTIVE_QUALITY:,}):")
    print(f"{'track':>7} {'random':>10} {'q(random)':>10} {'index-0':>10} {'q(index-0)':>11}")
    for track in sorted(MAINNET_MEDIAN):
        rows = by_track[track]
        rnd = mean([r["random"] for r in rows])
        zero = mean([r["zero"] for r in rows])
        print(
            f"{track:>7} {rnd:>10.6f} {quality(rnd):>10} "
            f"{zero:>10.6f} {quality(zero):>11}"
        )

    # The load-bearing diagnostic. On mainnet the entire field within a track
    # spans ~80-110 quality units, so if one nonce differs from another by much
    # more than that, quality measures luck rather than algorithm strength.
    print("\nNonce-to-nonce spread in achievable quality, against the mainnet")
    print("within-track spread of the whole competitive field:")
    print(f"{'track':>7} {'n':>3} {'spread':>8} {'mainnet':>8} {'ratio':>7}")
    for track in sorted(MAINNET_MEDIAN):
        rows = by_track[track]
        qualities = [quality(r["optimal"]) for r in rows]
        spread = max(qualities) - min(qualities)
        mainnet = MAINNET_WITHIN_TRACK_SPREAD[track]
        print(f"{track:>7} {len(rows):>3} {spread:>8} {mainnet:>8} {spread / mainnet:>6.1f}x")


if __name__ == "__main__":
    main()
