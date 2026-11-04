"""analyze.py - clean the collected vacancies and produce the evidence for Objectives 1-3 (+ RQ3).

Input : data/raw/vacancies.jsonl            (written by collect.py)
Output: outputs/tables/*.csv, outputs/figures/top_technologies.png,
        outputs/cleaning_funnel.csv, outputs/validation_sample.csv

Usage:
    python analyze.py                         # full pipeline
    python analyze.py --validate annotated.csv   # precision/recall of the dictionary
    python analyze.py --spotcheck outputs/title_spotcheck.csv   # check of the title inclusion rule
"""
import argparse
import html
import json
import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

RAW = pathlib.Path("data/raw/vacancies.jsonl")
OUT = pathlib.Path("outputs")
(OUT / "tables").mkdir(parents=True, exist_ok=True)
(OUT / "figures").mkdir(parents=True, exist_ok=True)
SEED = 42
MIN_GROUP = 30                    # smallest group that is compared statistically
DICTIONARY_VERSION = "v3"   # v2: "js" not inside Node.js/Vue.js, REST API needs "api"/"restful"; v3: stricter role rules + title inclusion rule

# ---------------------------------------------------------------- dictionaries
# name -> (regex applied to the lower-cased text, skills_only)
# skills_only=True: ambiguous token, matched only inside the structured key_skills field.
TECH = {
    "Python": (r"(?<![\w+#])python(?!\w)", False),
    "Java": (r"(?<![\w+#.])java(?![\w])", False),
    "JavaScript": (r"(?<![\w.])(javascript|js)(?!\w)", False),
    "TypeScript": (r"(?<!\w)(typescript|ts)(?!\w)", False),
    "C#": (r"(?<!\w)c#", False),
    "C++": (r"(?<!\w)c\+\+", False),
    "Go": (r"(?<!\w)(golang|go)(?!\w)", True),
    "PHP": (r"(?<!\w)php(?!\w)", False),
    "Kotlin": (r"(?<!\w)kotlin(?!\w)", False),
    "Swift": (r"(?<!\w)swift(?!\w)", True),
    "1C": (r"(?<!\w)(1с|1c)(?!\w)", False),
    "SQL": (r"(?<!\w)(sql|postgresql|mysql|mssql|oracle db)(?!\w)", False),
    "PostgreSQL": (r"(?<!\w)(postgresql|postgres)(?!\w)", False),
    "MongoDB": (r"(?<!\w)mongo(db)?(?!\w)", False),
    "Redis": (r"(?<!\w)redis(?!\w)", False),
    "React": (r"(?<!\w)react(\.js|js)?(?!\w)", False),
    "Vue": (r"(?<!\w)vue(\.js|js)?(?!\w)", False),
    "Angular": (r"(?<!\w)angular(?!\w)", False),
    "Node.js": (r"(?<!\w)node(\.js|js)?(?!\w)", False),
    "Django": (r"(?<!\w)django(?!\w)", False),
    "FastAPI": (r"(?<!\w)fastapi(?!\w)", False),
    "Spring": (r"(?<!\w)spring(?!\w)", False),
    ".NET": (r"(?<!\w)(\.net|asp\.net|dotnet)(?!\w)", False),
    "Docker": (r"(?<!\w)docker(?!\w)", False),
    "Kubernetes": (r"(?<!\w)(kubernetes|k8s)(?!\w)", False),
    "Git": (r"(?<!\w)git(lab|hub)?(?!\w)", False),
    "Linux": (r"(?<!\w)linux(?!\w)", False),
    "AWS": (r"(?<!\w)aws(?!\w)", False),
    "CI/CD": (r"(?<!\w)(ci/cd|ci-cd|jenkins)(?!\w)", False),
    "Kafka": (r"(?<!\w)kafka(?!\w)", False),
    "REST API": (r"(?<!\w)(restful|rest[ -]?api)(?!\w)", False),
    "Selenium": (r"(?<!\w)selenium(?!\w)", False),
    "Pandas": (r"(?<!\w)pandas(?!\w)", False),
    "TensorFlow/PyTorch": (r"(?<!\w)(tensorflow|pytorch)(?!\w)", False),
    "Power BI": (r"(?<!\w)power ?bi(?!\w)", False),
    "Flutter": (r"(?<!\w)flutter(?!\w)", False),
}

# role group from the vacancy title (first match wins); order matters.
# v3: word boundaries, no bare "engineer"/"инженер" (support, security and network engineers are not software roles).
# Inclusion rule: a posting is software-related if its title matches ANY group below; "Other" is excluded from the
# main analysis and kept only for the sensitivity analysis on all IT-category postings.
ROLE_RULES = [
    ("QA", r"\bqa\b|qa-|тестиров|tester|quality assurance|sdet|автотест"),
    ("Data/ML", r"\bdata\b|данн|machine learning|\bml\b|\bai\b|\bbi\b|power ?bi|нейро|\betl\b"),
    ("DevOps", r"devops|devsecops|mlops|\bsre\b|platform engineer|cloud engineer"),
    ("Mobile", r"android|\bios\b|mobile|мобильн|flutter|kotlin|swift"),
    ("Fullstack", r"full.?stack|фулл.?стек"),
    ("Frontend", r"front.?end|фронт|react|vue|angular|верстальщик|javascript|typescript|(?<![\w.])js\b"),
    ("Backend", r"back.?end|бэк|бек|python|\bjava\b|php|golang|c#|\.net|node|c\+\+|ruby|scala|rust|"
                r"разработчик|developer|программист|software engineer"),
]
# titles that are not software roles even if hh files them under IT
EXCLUDE_TITLE = (r"дизайнер|designer|маркетолог|marketing|менеджер|manager|руководител|начальник|директор|head of|"
                 r"sales|администратор|administrator|sysadmin|сетевой инженер|поддержк|helpdesk|help desk|"
                 r"оператор|data entry|ввод\w* данных|\bseo\b")

EXPERIENCE_LEVEL = {
    "noExperience": "junior",
    "between1And3": "junior",
    "between3And6": "middle",
    "moreThan6": "senior",
}


# ---------------------------------------------------------------- helpers
def strip_html(s):
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s))).strip()


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    a = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - a) / d, (c + a) / d


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def cramers_v(table):
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    k = min(table.shape) - 1
    return np.sqrt(chi2 / (n * k)) if n and k else np.nan


def role_group(title):
    t = (title or "").lower() + " "
    for name, pat in ROLE_RULES:
        if re.search(pat, t):
            return name
    return "Other"


def salary_kzt(row):
    """Monthly salary midpoint in KZT; None if not stated or other currency."""
    s = row.get("salary_range") or row.get("salary")
    if not isinstance(s, dict):
        return None, False, None
    cur = s.get("currency")
    lo, hi = s.get("from"), s.get("to")
    vals = [x for x in (lo, hi) if x]
    mid = float(np.mean(vals)) if vals else None
    return (mid if cur in ("KZT", "KZT ") else None), True, cur


# ---------------------------------------------------------------- pipeline
def load():
    with RAW.open(encoding="utf-8") as f:
        return pd.DataFrame(json.loads(line) for line in f)


FUNNEL = []
EXCLUDED = {}          # rows removed by EXCLUDE_TITLE (kept for the title spot-check)


def write_funnel():
    pd.DataFrame(FUNNEL, columns=["step", "n"]).to_csv(OUT / "cleaning_funnel.csv", index=False)


def clean(df):
    funnel = [("raw records", len(df))]
    df = df.drop_duplicates("id")
    funnel.append(("unique vacancy ids", len(df)))
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df = df[df["published_at"] >= pd.Timestamp("2026-01-01", tz="UTC")]
    funnel.append(("published in 2026", len(df)))

    df["desc_text"] = df["description"].map(strip_html)
    df = df[df["desc_text"].str.len() > 0]
    funnel.append(("non-empty description", len(df)))

    hit = df["name"].fillna("").str.lower().str.contains(EXCLUDE_TITLE, regex=True)
    EXCLUDED["list"] = df[hit]
    df = df[~hit]
    funnel.append(("non-software exclusion list (design, sales, helpdesk, ...)", len(df)))

    # reposts: same employer + same normalised title + same description text
    key = (df["employer_hash"].fillna("") + "|" + df["name"].fillna("").str.lower().str.strip()
           + "|" + df["desc_text"].str.lower().map(hash).astype(str))
    df = df.assign(_key=key).sort_values("published_at").drop_duplicates("_key").drop(columns="_key")
    funnel.append(("after repost removal", len(df)))
    FUNNEL[:] = funnel
    write_funnel()
    return df.reset_index(drop=True)


def label(df):
    df = df.copy()
    skills = df["key_skills"].map(lambda xs: {x.lower() for x in xs} if isinstance(xs, list) else set())
    text = (df["name"].fillna("") + " " + df["desc_text"]).str.lower()
    for tech, (pat, skills_only) in TECH.items():
        rx = re.compile(pat)
        in_skills = skills.map(lambda s: any(rx.fullmatch(x) or rx.search(x) for x in s))
        df[f"t_{tech}"] = in_skills if skills_only else (in_skills | text.map(lambda t: bool(rx.search(t))))
    df["role_group"] = df["name"].map(role_group)
    df["level"] = df["experience"].map(EXPERIENCE_LEVEL)
    sal = df.apply(salary_kzt, axis=1, result_type="expand")
    df["salary_kzt_mid"], df["salary_stated"], df["salary_currency"] = sal[0], sal[1], sal[2]
    df["remote"] = df["schedule"].eq("remote")
    return df


def tech_cols(df):
    return [c for c in df.columns if c.startswith("t_")]


def frequencies(df, tag="", title_note=""):
    n = len(df)
    rows = []
    for c in tech_cols(df):
        k = int(df[c].sum())
        lo, hi = wilson(k, n)
        rows.append({"technology": c[2:], "count": k, "share_pct": 100 * k / n,
                     "ci95_low": 100 * lo, "ci95_high": 100 * hi})
    out = pd.DataFrame(rows).sort_values("count", ascending=False)
    out.to_csv(OUT / "tables" / f"tech_frequencies{tag}.csv", index=False)
    top = out.head(15).iloc[::-1]
    plt.figure(figsize=(7, 5))
    plt.barh(top["technology"], top["share_pct"],
             xerr=[(top["share_pct"] - top["ci95_low"]).clip(lower=0), (top["ci95_high"] - top["share_pct"]).clip(lower=0)])
    plt.xlabel(f"% of vacancies (n = {n}), 95% Wilson CI")
    plt.title("Most frequently required technologies" + title_note)
    plt.tight_layout()
    plt.savefig(OUT / "figures" / f"top_technologies{tag}.png", dpi=200)
    plt.close()
    return out


def compare(df, group_col, top_n=10, tag=""):
    """Technology x group shares + chi-square (Holm-adjusted) for the top technologies."""
    top = frequencies_top(df, top_n)
    sizes = df[group_col].value_counts()
    keep = sizes[sizes >= MIN_GROUP].index
    sub = df[df[group_col].isin(keep)]
    shares = pd.DataFrame({g: [100 * sub.loc[sub[group_col] == g, f"t_{t}"].mean() for t in top]
                           for g in keep}, index=top)
    res, pvals = [], []
    for t in top:
        table = pd.crosstab(sub[group_col], sub[f"t_{t}"])
        if table.shape[1] < 2:
            pvals.append(np.nan)
            res.append((t, np.nan, np.nan))
            continue
        chi2, p, _, _ = stats.chi2_contingency(table)
        pvals.append(p)
        res.append((t, chi2, cramers_v(table)))
    adj = holm(np.nan_to_num(pvals, nan=1.0))
    test = pd.DataFrame(res, columns=["technology", "chi2", "cramers_v"])
    test["p"], test["p_holm"] = pvals, adj
    shares.round(1).to_csv(OUT / "tables" / f"tech_share_by_{tag or group_col}.csv")
    test.to_csv(OUT / "tables" / f"tech_tests_by_{tag or group_col}.csv", index=False)
    return shares, test


def frequencies_top(df, n):
    counts = {c[2:]: df[c].sum() for c in tech_cols(df)}
    return [k for k, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:n]]


def salary_analysis(df, top_n=10):
    top = frequencies_top(df, top_n)
    rows = []
    for t in top:
        has = df[f"t_{t}"]
        stated_with, stated_without = df.loc[has, "salary_stated"].mean(), df.loc[~has, "salary_stated"].mean()
        table = pd.crosstab(has, df["salary_stated"])
        p_stated = stats.chi2_contingency(table)[1] if table.shape == (2, 2) else np.nan
        a = df.loc[has, "salary_kzt_mid"].dropna()
        b = df.loc[~has, "salary_kzt_mid"].dropna()
        p_sal = stats.mannwhitneyu(a, b).pvalue if len(a) >= MIN_GROUP and len(b) >= MIN_GROUP else np.nan
        rows.append({"technology": t, "pct_salary_stated_with": 100 * stated_with,
                     "pct_salary_stated_without": 100 * stated_without, "p_stated": p_stated,
                     "n_kzt_with": len(a), "median_kzt_with": a.median(),
                     "n_kzt_without": len(b), "median_kzt_without": b.median(), "p_salary": p_sal})
    out = pd.DataFrame(rows)
    out["p_stated_holm"] = holm(np.nan_to_num(out["p_stated"], nan=1.0))
    out["p_salary_holm"] = holm(np.nan_to_num(out["p_salary"], nan=1.0))
    out.to_csv(OUT / "tables" / "salary_by_technology.csv", index=False)
    return out


def validation_sample(df, size=100):
    s = df.sample(min(size, len(df)), random_state=SEED)
    tc = tech_cols(df)
    pd.DataFrame({
        "id": s["id"],
        "text": (s["name"].fillna("") + " | " + s["key_skills"].map(lambda x: ", ".join(x) if isinstance(x, list) else "")
                 + " | " + s["desc_text"].str[:1500]),
        "detected": s[tc].apply(lambda r: ";".join(c[2:] for c in tc if r[c]), axis=1),
        "true_techs": "",         # fill in by hand: technologies really required, separated by ";"
    }).to_csv(OUT / "validation_sample.csv", index=False)


def validate(path):
    a = pd.read_csv(path).fillna("")
    split = lambda s: {x.strip() for x in str(s).split(";") if x.strip()}
    tp = fp = fn = 0
    for det, true in zip(a["detected"], a["true_techs"]):
        d, t = split(det), split(true)
        tp, fp, fn = tp + len(d & t), fp + len(d - t), fn + len(t - d)
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else np.nan
    print(f"micro precision={prec:.3f} recall={rec:.3f} F1={f1:.3f} (tp={tp}, fp={fp}, fn={fn})")


def title_spotcheck(df_it, size=40):
    """Hand check of the title inclusion rule: 40 titles from each of three strata
    (included / not matched by any role rule / removed by the exclusion list)."""
    strata = ((df_it[df_it["role_group"] != "Other"], "included"),
              (df_it[df_it["role_group"] == "Other"], "other"),
              (EXCLUDED.get("list", df_it.iloc[0:0]), "list_excluded"))
    parts = []
    for part, name in strata:
        if len(part) == 0:
            continue
        smp = part.sample(min(size, len(part)), random_state=SEED)
        parts.append(pd.DataFrame({"title": smp["name"], "stratum": name, "stratum_n": len(part),
                                   "is_software": ""}))     # fill is_software by hand: 1 = software role, 0 = not
    pd.concat(parts).sample(frac=1, random_state=SEED).to_csv(OUT / "title_spotcheck.csv", index=False)


def spotcheck(path):
    a = pd.read_csv(path).fillna("")
    a = a[a["is_software"].astype(str).str.strip() != ""].copy()
    a["is_software"] = a["is_software"].astype(int)
    est = {}
    for name, g in a.groupby("stratum"):
        k, m, n_st = int(g["is_software"].sum()), len(g), int(g["stratum_n"].iloc[0])
        est[name] = (k, m, n_st)
        print(f"{name:14s}: {k}/{m} judged to be software roles ({k / m:.3f}); stratum size {n_st}")
    if {"included", "other", "list_excluded"} <= set(est):
        k, m, n_st = est["included"]
        kept_true = k / m * n_st
        lost = sum(est[x][0] / est[x][1] * est[x][2] for x in ("other", "list_excluded"))
        print(f"estimated precision of the inclusion rule: {k / m:.3f}")
        print(f"estimated recall of the inclusion rule   : {kept_true / (kept_true + lost):.3f} (point estimate, small samples)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", help="CSV with the hand-filled true_techs column")
    ap.add_argument("--spotcheck", help="CSV (title_spotcheck.csv) with the hand-filled is_software column")
    args = ap.parse_args()
    if args.validate:
        validate(args.validate)
        return
    if args.spotcheck:
        spotcheck(args.spotcheck)
        return

    df_it = label(clean(load()))                                   # all IT-category postings after cleaning
    df = df_it[df_it["role_group"] != "Other"].reset_index(drop=True)   # software-related title = analysis population
    FUNNEL.append(("software-related title (role rules) = analysis population", len(df)))
    write_funnel()
    df.drop(columns=["description"]).to_csv(OUT / "processed_vacancies.csv", index=False)
    print(pd.read_csv(OUT / "cleaning_funnel.csv").to_string(index=False))

    print("\nShare of the analysis population (%):")
    print(f"  experience missing : {100 * df['experience'].isna().mean():.1f}")
    print(f"  schedule missing   : {100 * df['schedule'].isna().mean():.1f}")
    print(f"  area missing       : {100 * df['area'].isna().mean():.1f}")
    print(f"  key skills empty   : {100 * df['key_skills'].map(lambda x: not isinstance(x, list) or len(x) == 0).mean():.1f}")
    print(f"  salary stated      : {100 * df['salary_stated'].mean():.1f}")
    print("\nTop technologies (analysis population, n = %d):" % len(df))
    print(frequencies(df).head(15).round(1).to_string(index=False))
    print("\nSensitivity: all IT-category postings after cleaning (n = %d), top 10:" % len(df_it))
    print(frequencies(df_it, "_all_it", " (all IT-category postings)").head(10).round(1).to_string(index=False))
    print("\nRole groups (analysis population):", df["role_group"].value_counts().to_dict())
    print("Role groups (all IT-category, incl. Other):", df_it["role_group"].value_counts().to_dict())
    print("Levels:", df["level"].value_counts(dropna=False).to_dict())
    compare(df, "role_group")
    compare(df, "level")
    salary_analysis(df)
    validation_sample(df)
    title_spotcheck(df_it)
    print("\nTables and figures written to ./outputs")


if __name__ == "__main__":
    main()
