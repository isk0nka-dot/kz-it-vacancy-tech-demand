"""collect.py - collect software-related vacancy snapshots for Kazakhstan from the hh API.

Design notes (see Assignment 3, Parts 2, 4 and 6):
  * only public vacancy data is requested; contact fields are never stored;
  * the employer id is stored as a SHA-256 hash (needed only to detect reposts);
  * requests are rate-limited (PAUSE seconds) and retried with back-off;
  * the API returns at most 2000 results per query, so every query covers ONE day
    and is split by professional role if a day is close to the cap;
  * new IDs are appended to data/raw/vacancies.jsonl; existing IDs are skipped,
    so the script can be run every week (snapshots) without duplicating work.

Usage:
    export HH_USER_AGENT="KZ-IT-demand-study/1.0 (your_email@example.com)"
    # optional, only if the API asks for authorisation:
    # export HH_TOKEN="..."
    python collect.py --days 30
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import time

import requests

API = "https://api.hh.ru"
AREA_KZ = 40                      # hh area id for Kazakhstan
PAUSE = 0.5                       # seconds between requests (low request rate)
PER_PAGE = 100
RESULT_CAP = 2000                 # API limit: page * per_page <= 2000
UA = os.environ.get("HH_USER_AGENT", "KZ-IT-demand-study/1.0 (student research)")
TOKEN = os.environ.get("HH_TOKEN")

RAW = pathlib.Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)
STORE = RAW / "vacancies.jsonl"
ROLES_FILE = RAW / "it_roles.json"


def get(path, params=None):
    headers = {"User-Agent": UA, "HH-User-Agent": UA}
    if TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"
    for attempt in range(8):
        try:
            r = requests.get(API + path, params=params, headers=headers, timeout=30)
        except requests.exceptions.RequestException as e:      # timeouts, dropped connections
            wait = min(2 ** attempt * 2, 60)
            print(f"Network error on {path} ({type(e).__name__}); retry {attempt + 1}/8 in {wait}s")
            time.sleep(wait)
            continue
        if r.status_code == 200:
            time.sleep(PAUSE)
            return r.json()
        if r.status_code in (429, 500, 502, 503):
            time.sleep(2 ** attempt * 2)
            continue
        print(f"HTTP {r.status_code} for {path}: {r.text[:300]}")
        r.raise_for_status()
    raise RuntimeError(f"Request failed after retries: {path} {params}")


def it_roles():
    """IDs of the roles in hh's Information Technology category (read from the API)."""
    data = get("/professional_roles")
    roles = {}
    for cat in data["categories"]:
        if "информационные технологии" in cat["name"].lower() or "information technology" in cat["name"].lower():
            for r in cat["roles"]:
                roles[r["id"]] = r["name"]
    if not roles:
        raise RuntimeError("IT category not found in /professional_roles - check the response manually")
    ROLES_FILE.write_text(json.dumps(roles, ensure_ascii=False, indent=2), encoding="utf-8")
    return roles


def search_window(day, role_ids):
    """All vacancy ids published on `day` for the given role ids."""
    ids = []
    page = 0
    while True:
        params = {
            "host": "hh.kz",
            "area": AREA_KZ,
            "professional_role": role_ids,
            "date_from": f"{day}T00:00:00",
            "date_to": f"{day}T23:59:59",
            "per_page": PER_PAGE,
            "page": page,
            "order_by": "publication_time",
        }
        data = get("/vacancies", params)
        ids += [item["id"] for item in data["items"]]
        page += 1
        if page >= data["pages"] or page * PER_PAGE >= RESULT_CAP:
            break
    return ids, data.get("found", 0)


def collect_day(day, role_ids):
    ids, found = search_window(day, role_ids)
    if found > RESULT_CAP - 100 and len(role_ids) > 1:     # close to the cap -> split by role
        ids = []
        for rid in role_ids:
            sub, _ = search_window(day, [rid])
            ids += sub
    return set(ids)


def slim(v):
    """Keep only the fields needed for the analysis (data minimisation)."""
    emp = (v.get("employer") or {}).get("id")
    return {
        "id": v["id"],
        "name": v.get("name"),
        "published_at": v.get("published_at"),
        "area": (v.get("area") or {}).get("name"),
        "experience": (v.get("experience") or {}).get("id"),
        "schedule": (v.get("schedule") or {}).get("id"),
        "employment": (v.get("employment") or {}).get("id"),
        "salary": v.get("salary"),
        "salary_range": v.get("salary_range"),
        "key_skills": [k["name"] for k in v.get("key_skills", [])],
        "roles": [r["name"] for r in v.get("professional_roles", [])],
        "description": v.get("description"),
        "employer_hash": hashlib.sha256(str(emp).encode()).hexdigest()[:16] if emp else None,
        "collected_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }


def seen_ids():
    if not STORE.exists():
        return set()
    with STORE.open(encoding="utf-8") as f:
        return {json.loads(line)["id"] for line in f}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30, help="how many past days to query")
    ap.add_argument("--since", default="2026-01-01", help="never query before this date")
    args = ap.parse_args()

    roles = it_roles()
    role_ids = list(roles)
    print(f"{len(role_ids)} IT roles loaded")

    today = dt.date.today()
    start = max(dt.date.fromisoformat(args.since), today - dt.timedelta(days=args.days))
    seen = seen_ids()
    new_total = 0
    day = start
    while day <= today:
        ids = collect_day(day.isoformat(), role_ids)
        fresh = [i for i in ids if i not in seen]
        with STORE.open("a", encoding="utf-8") as f:
            for i in fresh:
                f.write(json.dumps(slim(get(f"/vacancies/{i}", {"host": "hh.kz"})), ensure_ascii=False) + "\n")
                seen.add(i)
        new_total += len(fresh)
        print(f"{day}: {len(ids)} found, {len(fresh)} new")
        day += dt.timedelta(days=1)
    print(f"Done. New vacancies stored: {new_total}. Total in store: {len(seen)}")


if __name__ == "__main__":
    main()