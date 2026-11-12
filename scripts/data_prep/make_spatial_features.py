"""Build per-location spatial features from the real ADM5 boundary geojson.

Outputs sector_spatial.csv with, per location id:
  centroid_lon, centroid_lat   -- polygon centroid (shoelace), static geography
  n_neighbours                 -- degree in the edge-adjacency graph
and writes sector_neighbours.json: {location_id: [neighbour ids...]}.

Adjacency: two polygons are neighbours if their boundaries share an edge,
detected as >=2 shared (rounded) vertices. No shapely dependency.
"""

import json
import os
from collections import defaultdict
from pathlib import Path

# Directory holding the boundary geojson and receiving the outputs (default: current directory).
DATA_DIR = Path(os.environ.get("AR_DATA_DIR", "."))

GEO = DATA_DIR / "geoBoundaries-RWA-ADM5.geojson"
ROUND = 6  # decimal places to snap vertices for shared-edge detection


def rings(geom):
    """Yield coordinate rings for Polygon / MultiPolygon."""
    if geom["type"] == "Polygon":
        for r in geom["coordinates"]:
            yield r
    elif geom["type"] == "MultiPolygon":
        for poly in geom["coordinates"]:
            for r in poly:
                yield r


def centroid(geom):
    """Area-weighted polygon centroid (shoelace), outer rings only."""
    cx = cy = area2 = 0.0
    for poly_idx, r in enumerate(rings(geom)):
        a = 0.0
        rx = ry = 0.0
        for (x0, y0), (x1, y1) in zip(r, r[1:]):
            cross = x0 * y1 - x1 * y0
            a += cross
            rx += (x0 + x1) * cross
            ry += (y0 + y1) * cross
        cx += rx
        cy += ry
        area2 += a
    if area2 == 0:
        # fall back to mean of vertices
        pts = [p for r in rings(geom) for p in r]
        return sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)
    return cx / (3 * area2), cy / (3 * area2)


def main():
    """Compute centroids and the edge-adjacency neighbour graph, then write outputs."""
    d = json.load(open(GEO))
    feats = d["features"]
    ids = [f["id"] for f in feats]

    # centroids
    cents = {f["id"]: centroid(f["geometry"]) for f in feats}

    # vertex -> set of polygon ids that touch it
    vert_to_ids = defaultdict(set)
    for f in feats:
        for r in rings(f["geometry"]):
            for x, y in r:
                vert_to_ids[(round(x, ROUND), round(y, ROUND))].add(f["id"])

    # count shared vertices per pair; adjacency if >= 2 (an edge)
    shared = defaultdict(int)
    for owners in vert_to_ids.values():
        ow = sorted(owners)
        for i in range(len(ow)):
            for j in range(i + 1, len(ow)):
                shared[(ow[i], ow[j])] += 1

    nbrs = defaultdict(list)
    for (a, b), c in shared.items():
        if c >= 2:
            nbrs[a].append(b)
            nbrs[b].append(a)

    # write neighbour graph
    json.dump({i: nbrs.get(i, []) for i in ids}, open(DATA_DIR / "sector_neighbours.json", "w"))

    # write centroid/degree table
    with open(DATA_DIR / "sector_spatial.csv", "w") as fh:
        fh.write("location,centroid_lon,centroid_lat,n_neighbours\n")
        for i in ids:
            lon, lat = cents[i]
            fh.write(f"{i},{lon:.6f},{lat:.6f},{len(nbrs.get(i, []))}\n")

    degs = [len(nbrs.get(i, [])) for i in ids]
    iso = sum(1 for x in degs if x == 0)
    print(
        f"{len(ids)} units; degree min/mean/max = {min(degs)}/{sum(degs) / len(degs):.1f}/{max(degs)}; isolated={iso}"
    )


if __name__ == "__main__":
    main()
