"""Debug: plot 4-nearest-neighbor distances for the panos in output/panos.json.

Useful for sanity-checking density/coverage of the snapped Street-View grid.
NOT part of the agent runtime.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402


def main() -> int:
    if not config.PANOS_PATH.exists():
        print(f"{config.PANOS_PATH} not found; run the agent first.")
        return 1

    panos = json.loads(config.PANOS_PATH.read_text())
    if len(panos) < 5:
        print("Need at least 5 panos to plot.")
        return 1

    lats = np.array([p["lat"] for p in panos])
    lngs = np.array([p["lng"] for p in panos])
    lng_factor = np.cos(np.radians(lats.mean()))
    lat_m = lats[:, None] - lats[None, :]
    lng_m = lngs[:, None] - lngs[None, :]
    dist_matrix = np.sqrt((lat_m * 111320) ** 2 + (lng_m * 111320 * lng_factor) ** 2)
    np.fill_diagonal(dist_matrix, np.inf)
    nearest_4 = np.sort(dist_matrix, axis=1)[:, :4]
    distances = nearest_4.flatten()

    plt.figure(figsize=(12, 5))
    plt.scatter(range(len(distances)), distances, alpha=0.95, s=2, color="steelblue")
    plt.xlabel("pair index")
    plt.ylabel("distance (meters)")
    plt.title(f"4 nearest neighbor distances — {len(distances):,} pairs")
    plt.tight_layout()

    out_path = config.OUTPUT_DIR / "neighbor_dist.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    print(
        f"min: {distances.min():.1f}m | max: {distances.max():.1f}m | "
        f"mean: {distances.mean():.1f}m | median: {np.median(distances):.1f}m"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
