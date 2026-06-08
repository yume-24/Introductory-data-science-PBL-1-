#!/usr/bin/env python3
"""
Compute walk, drive, and public-transit travel times from HDB flats to nearest POIs
using the free Singapore OneMap Routing API.

Requires ONEMAP_EMAIL and ONEMAP_PASSWORD in .env or environment.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parent
CACHE_DIR = BASE / "cache"
INPUT_CSV = BASE / "hdb_resale_prices_final.csv"
POI_COORDS_CSV = CACHE_DIR / "poi_coords.csv"
OD_UNIQUE_CSV = CACHE_DIR / "od_unique.csv"
ROUTES_JSONL = CACHE_DIR / "onemap_routes.jsonl"
OUTPUT_CSV = BASE / "hdb_resale_prices_with_travel_times.csv"

AUTH_URL = "https://www.onemap.gov.sg/api/auth/post/getToken"
SEARCH_URL = "https://www.onemap.gov.sg/api/common/elastic/search"
ROUTE_URL = "https://www.onemap.gov.sg/api/public/routingsvc/route"

# Fixed departure for reproducible transit times (weekday morning peak)
TRANSIT_DATE = "06-01-2026"  # MM-DD-YYYY
TRANSIT_TIME = "08:30:00"

POI_COLUMNS = {
    "mrt": "nearest_mrt_exit",
    "hawker": "nearest_hawker",
    "activesg": "nearest_activesg",
    "park": "nearest_park",
}


@dataclass
class OneMapClient:
    email: str
    password: str
    token: str | None = None
    expiry: int = 0
    session: Any = None

    def __post_init__(self) -> None:
        import requests

        self.session = requests.Session()

    def refresh_token(self) -> str:
        r = self.session.post(
            AUTH_URL,
            json={"email": self.email, "password": self.password},
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        self.token = data["access_token"]
        self.expiry = int(data["expiry_timestamp"])
        return self.token

    def ensure_token(self) -> str:
        if self.token and time.time() < self.expiry - 60:
            return self.token
        return self.refresh_token()

    def _headers(self) -> dict[str, str]:
        return {"Authorization": self.ensure_token()}

    def search(self, query: str, page: int = 1) -> dict[str, Any]:
        params = {
            "searchVal": query,
            "returnGeom": "Y",
            "getAddrDetails": "Y",
            "pageNum": page,
        }
        r = self.session.get(
            SEARCH_URL, params=params, headers=self._headers(), timeout=30
        )
        if r.status_code == 429:
            time.sleep(5)
            r = self.session.get(
                SEARCH_URL, params=params, headers=self._headers(), timeout=30
            )
        r.raise_for_status()
        return r.json()

    def route(
        self,
        origin_lat: float,
        origin_lon: float,
        dest_lat: float,
        dest_lon: float,
        route_type: str,
    ) -> dict[str, Any] | None:
        params: dict[str, str] = {
            "start": f"{origin_lat},{origin_lon}",
            "end": f"{dest_lat},{dest_lon}",
            "routeType": route_type,
        }
        if route_type == "pt":
            params.update(
                {
                    "date": TRANSIT_DATE,
                    "time": TRANSIT_TIME,
                    "mode": "TRANSIT",
                    "numItineraries": "1",
                }
            )
        import requests

        retryable = {429, 500, 502, 503, 504}
        # Transit routing is slower; allow longer read timeout
        timeout_s = 120 if route_type == "pt" else 90
        for attempt in range(10):
            try:
                r = self.session.get(
                    ROUTE_URL,
                    params=params,
                    headers=self._headers(),
                    timeout=timeout_s,
                )
            except (requests.Timeout, requests.ConnectionError) as e:
                wait = min(120, 2**attempt * 3)
                print(
                    f"  network error ({type(e).__name__}), retry in {wait}s "
                    f"(attempt {attempt + 1}/10)...",
                    flush=True,
                )
                time.sleep(wait)
                continue
            if r.status_code in retryable:
                wait = min(120, 2**attempt * 3)
                print(
                    f"  HTTP {r.status_code}, retry in {wait}s "
                    f"(attempt {attempt + 1}/10)...",
                    flush=True,
                )
                time.sleep(wait)
                continue
            if r.status_code == 401:
                self.refresh_token()
                continue
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        return None


def round6(x: float) -> float:
    return round(float(x), 6)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def geocode_query_variants(name: str) -> list[str]:
    """Build search strings to try (OneMap often fails with ', Singapore')."""
    variants: list[str] = [name.strip()]
    no_paren = re.sub(r"\s*\([^)]*\)", "", name).strip()
    if no_paren and no_paren not in variants:
        variants.append(no_paren)

    if "@" in name:
        after_at = name.split("@", 1)[1].strip()
        if after_at:
            variants.append(after_at)

    # Hawker names like "Ang Mo Kio Ave 10 Blk 409 (...)" → try block address part
    blk = re.search(r"(.+?\bBlk\s+\d+)", name, re.I)
    if blk:
        variants.append(blk.group(1).strip())

    # Normalise park abbreviations for search
    if " PK" in name.upper() or name.upper().endswith(" PK"):
        expanded = re.sub(r"\bPK\b", "PARK", name, flags=re.I)
        if expanded not in variants:
            variants.append(expanded)

    # Deduplicate while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for v in variants:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def score_search_result(name: str, result: dict[str, Any]) -> float:
    """Higher = better match for POI name."""
    target = _norm(name)
    fields = " ".join(
        str(result.get(k, "") or "")
        for k in ("SEARCHVAL", "BUILDING", "ADDRESS", "ROAD_NAME")
    )
    field_norm = _norm(fields)
    if not target or not field_norm:
        return 0.0

    target_tokens = set(target.split())
    field_tokens = set(field_norm.split())
    overlap = len(target_tokens & field_tokens) / max(len(target_tokens), 1)

    score = overlap
    if target in field_norm or field_norm in target:
        score += 0.5
    if "mrt station" in target and "mrt station" in field_norm:
        score += 0.3
    if "mrt station" in target and "mrt station" not in field_norm:
        score -= 0.4
    return score


def pick_best_result(name: str, results: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not results:
        return None
    return max(results, key=lambda r: score_search_result(name, r))


def geocode_one_poi(client: OneMapClient, name: str) -> tuple[float, float] | None:
    best: dict[str, Any] | None = None
    best_score = -1.0
    for query in geocode_query_variants(name):
        data = client.search(query)
        results = data.get("results") or []
        if not results:
            continue
        candidate = pick_best_result(name, results)
        if not candidate:
            continue
        s = score_search_result(name, candidate)
        if s > best_score:
            best_score = s
            best = candidate
        if s >= 0.8:
            break

    if not best or best_score < 0.25:
        return None
    return float(best["LATITUDE"]), float(best["LONGITUDE"])


def build_od_unique() -> int:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    seen_od: set[tuple] = set()
    rows_out: list[dict] = []

    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lat6 = round6(row["LATITUDE"])
            lon6 = round6(row["LONGITUDE"])
            for poi_type, col in POI_COLUMNS.items():
                dest = row[col].strip()
                key = (lat6, lon6, poi_type, dest)
                if key in seen_od:
                    continue
                seen_od.add(key)
                rows_out.append(
                    {
                        "lat6": lat6,
                        "lon6": lon6,
                        "poi_type": poi_type,
                        "dest_name": dest,
                    }
                )

    with open(OD_UNIQUE_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["lat6", "lon6", "poi_type", "dest_name"]
        )
        w.writeheader()
        w.writerows(rows_out)
    return len(rows_out)


def geocode_pois(client: OneMapClient, limit: int | None = None) -> None:
    import requests

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    existing: dict[str, tuple[float, float]] = {}
    if POI_COORDS_CSV.exists():
        with open(POI_COORDS_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["dest_lat"] and row["dest_lon"]:
                    existing[row["dest_name"]] = (
                        float(row["dest_lat"]),
                        float(row["dest_lon"]),
                    )

    names: set[str] = set()
    with open(OD_UNIQUE_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            names.add(row["dest_name"])

    todo = sorted(names - existing.keys())
    if limit:
        todo = todo[:limit]
    print(f"Geocoding {len(todo)} POI names ({len(existing)} already cached)")

    failed: list[str] = []
    for i, name in enumerate(todo, 1):
        try:
            coords = geocode_one_poi(client, name)
        except requests.HTTPError as e:
            print(f"  [{i}/{len(todo)}] search failed for {name!r}: {e}")
            failed.append(name)
            time.sleep(0.25)
            continue

        if coords is None:
            print(f"  [{i}/{len(todo)}] no results for {name!r}")
            failed.append(name)
        else:
            existing[name] = coords
            if i % 20 == 0 or i == len(todo):
                print(f"  [{i}/{len(todo)}] geocoded {name!r}")

        time.sleep(0.25)

    if failed:
        fail_path = CACHE_DIR / "geocode_failed.txt"
        fail_path.write_text("\n".join(failed) + "\n", encoding="utf-8")
        print(f"  {len(failed)} POIs not geocoded (listed in {fail_path})")

    with open(POI_COORDS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["dest_name", "dest_lat", "dest_lon"]
        )
        w.writeheader()
        for name in sorted(existing.keys()):
            lat, lon = existing[name]
            w.writerow(
                {"dest_name": name, "dest_lat": lat, "dest_lon": lon}
            )


def attach_dest_coords() -> None:
    poi: dict[str, tuple[float, float]] = {}
    if POI_COORDS_CSV.exists():
        with open(POI_COORDS_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["dest_lat"] and row["dest_lon"]:
                    poi[row["dest_name"]] = (
                        float(row["dest_lat"]),
                        float(row["dest_lon"]),
                    )

    rows: list[dict] = []
    with open(OD_UNIQUE_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            base = {
                "lat6": row["lat6"],
                "lon6": row["lon6"],
                "poi_type": row["poi_type"],
                "dest_name": row["dest_name"],
            }
            dest = base["dest_name"]
            if dest in poi:
                dlat, dlon = poi[dest]
                rows.append({**base, "dest_lat": dlat, "dest_lon": dlon})
            else:
                rows.append({**base, "dest_lat": "", "dest_lon": ""})

    fieldnames = [
        "lat6",
        "lon6",
        "poi_type",
        "dest_name",
        "dest_lat",
        "dest_lon",
    ]
    with open(OD_UNIQUE_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def parse_duration(data: dict[str, Any] | None, route_type: str) -> int | None:
    if not data:
        return None
    if route_type in ("walk", "drive"):
        summary = data.get("route_summary") or {}
        t = summary.get("total_time")
        return int(t) if t is not None else None
    if route_type == "pt":
        itineraries = (data.get("plan") or {}).get("itineraries") or []
        if not itineraries:
            return None
        d = itineraries[0].get("duration")
        return int(d) if d is not None else None
    return None


def load_route_cache() -> dict[tuple, dict[str, int | None]]:
    cache: dict[tuple, dict[str, int | None]] = {}
    if not ROUTES_JSONL.exists():
        return cache
    with open(ROUTES_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (
                rec["lat6"],
                rec["lon6"],
                rec["poi_type"],
                rec["dest_name"],
            )
            # Last line wins (resume may append duplicate keys)
            prev = cache.get(key, {})
            cache[key] = {
                "walk_sec": rec.get("walk_sec", prev.get("walk_sec")),
                "drive_sec": rec.get("drive_sec", prev.get("drive_sec")),
                "transit_sec": rec.get("transit_sec", prev.get("transit_sec")),
                "complete": rec.get("complete", prev.get("complete", False)),
            }
    return cache


def append_route_cache(rec: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(ROUTES_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def fetch_routes(
    client: OneMapClient,
    delay: float,
    limit: int | None = None,
) -> None:
    cache = load_route_cache()
    todo: list[dict] = []
    with open(OD_UNIQUE_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (
                float(row["lat6"]),
                float(row["lon6"]),
                row["poi_type"],
                row["dest_name"],
            )
            if cache.get(key, {}).get("complete"):
                continue
            if not row.get("dest_lat") or not row.get("dest_lon"):
                continue
            todo.append(row)

    if limit:
        todo = todo[:limit]
    total = len(todo)
    print(f"Routing {total} unique ODs (up to {total * 3} API calls)")

    for i, row in enumerate(todo, 1):
        lat6 = float(row["lat6"])
        lon6 = float(row["lon6"])
        dlat = float(row["dest_lat"])
        dlon = float(row["dest_lon"])
        key = (lat6, lon6, row["poi_type"], row["dest_name"])
        cached = cache.get(key, {})

        durations: dict[str, int | None] = {
            "walk_sec": cached.get("walk_sec"),
            "drive_sec": cached.get("drive_sec"),
            "transit_sec": cached.get("transit_sec"),
        }

        for route_type, field in [
            ("walk", "walk_sec"),
            ("drive", "drive_sec"),
            ("pt", "transit_sec"),
        ]:
            if durations[field] is not None:
                continue
            data = client.route(lat6, lon6, dlat, dlon, route_type)
            durations[field] = parse_duration(data, route_type)
            # Save after each mode so a crash mid-OD can resume
            partial = {
                "lat6": lat6,
                "lon6": lon6,
                "poi_type": row["poi_type"],
                "dest_name": row["dest_name"],
                "complete": False,
                **durations,
            }
            append_route_cache(partial)
            cache[key] = {**durations, "complete": False}
            time.sleep(delay)

        rec = {
            "lat6": lat6,
            "lon6": lon6,
            "poi_type": row["poi_type"],
            "dest_name": row["dest_name"],
            "complete": True,
            **durations,
        }
        append_route_cache(rec)
        cache[key] = {**durations, "complete": True}

        if i % 50 == 0 or i == total:
            print(f"  [{i}/{total}] {row['poi_type']} {row['dest_name'][:40]}")

        if i % 100 == 0:
            client.ensure_token()


def merge_to_output() -> None:
    cache = load_route_cache()
    lookup: dict[tuple, dict[str, float | None]] = {}
    for key, d in cache.items():
        poi_type = key[2]
        lookup[key] = {
            "walk_min": sec_to_min(d.get("walk_sec")),
            "drive_min": sec_to_min(d.get("drive_sec")),
            "transit_min": sec_to_min(d.get("transit_sec")),
        }

    with open(INPUT_CSV, newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        fieldnames = list(reader.fieldnames or [])
        new_cols = []
        for poi in POI_COLUMNS:
            for mode in ("walk", "drive", "transit"):
                col = f"duration_{poi}_{mode}_min"
                if col not in fieldnames:
                    new_cols.append(col)
        out_fields = fieldnames + new_cols

        rows_out = []
        for row in reader:
            lat6 = round6(row["LATITUDE"])
            lon6 = round6(row["LONGITUDE"])
            for poi_type, nearest_col in POI_COLUMNS.items():
                dest = row[nearest_col].strip()
                key = (lat6, lon6, poi_type, dest)
                vals = lookup.get(key, {})
                row[f"duration_{poi_type}_walk_min"] = fmt(vals.get("walk_min"))
                row[f"duration_{poi_type}_drive_min"] = fmt(vals.get("drive_min"))
                row[f"duration_{poi_type}_transit_min"] = fmt(
                    vals.get("transit_min")
                )
            rows_out.append(row)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fout:
        w = csv.DictWriter(fout, fieldnames=out_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows_out)
    print(f"Wrote {OUTPUT_CSV} ({len(rows_out)} rows)")


def sec_to_min(sec: int | None) -> float | None:
    if sec is None:
        return None
    return round(sec / 60.0, 2)


def fmt(v: float | None) -> str:
    if v is None:
        return ""
    return str(v)


def print_status() -> None:
    od_count = 0
    if OD_UNIQUE_CSV.exists():
        with open(OD_UNIQUE_CSV, encoding="utf-8") as f:
            od_count = sum(1 for _ in f) - 1

    poi_count = 0
    if POI_COORDS_CSV.exists():
        with open(POI_COORDS_CSV, encoding="utf-8") as f:
            poi_count = sum(1 for _ in f) - 1

    cache = load_route_cache()
    complete = sum(1 for v in cache.values() if v.get("complete"))
    print(f"Unique ODs:        {od_count}")
    print(f"POIs geocoded:     {poi_count}")
    print(f"ODs routed (done): {complete} / {od_count}")
    if complete and od_count:
        pct = 100.0 * complete / od_count
        print(f"Progress:          {pct:.1f}%")


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(BASE / ".env")
    except ImportError:
        pass
    parser = argparse.ArgumentParser(description="OneMap travel time batch job")
    parser.add_argument(
        "--step",
        choices=["all", "build", "geocode", "route", "merge", "status"],
        default="all",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit rows for geocode/route (testing)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.35,
        help="Seconds between routing API calls",
    )
    args = parser.parse_args()

    if args.step == "status":
        print_status()
        return

    steps = (
        ["build", "geocode", "route", "merge"]
        if args.step == "all"
        else [args.step]
    )

    if "build" in steps:
        n = build_od_unique()
        print(f"Built {OD_UNIQUE_CSV} with {n} unique ODs")

    email = os.environ.get("ONEMAP_EMAIL", "").strip()
    password = os.environ.get("ONEMAP_PASSWORD", "").strip()
    need_auth = any(s in steps for s in ("geocode", "route"))

    client: OneMapClient | None = None
    if need_auth:
        if not email or not password:
            print(
                "Error: set ONEMAP_EMAIL and ONEMAP_PASSWORD in .env\n"
                f"Copy {BASE / '.env.example'} to {BASE / '.env'}",
                file=sys.stderr,
            )
            sys.exit(1)
        client = OneMapClient(email=email, password=password)
        client.refresh_token()
        print("OneMap token obtained")

    if "geocode" in steps:
        assert client
        geocode_pois(client, limit=args.limit)
        attach_dest_coords()

    if "route" in steps:
        assert client
        if not OD_UNIQUE_CSV.exists():
            build_od_unique()
        with open(OD_UNIQUE_CSV, encoding="utf-8") as f:
            header = f.readline()
        if "dest_lat" not in header:
            if not POI_COORDS_CSV.exists():
                geocode_pois(client, limit=args.limit)
            attach_dest_coords()
        fetch_routes(client, delay=args.delay, limit=args.limit)

    if "merge" in steps:
        cache = load_route_cache()
        done = sum(1 for v in cache.values() if v.get("complete"))
        if "route" not in steps and done < 39000:
            print(
                f"Warning: only {done} ODs fully routed; "
                "run --step route to finish before merging."
            )
        merge_to_output()


if __name__ == "__main__":
    main()
