"""extra_analysis.py - additional analyses for Assignment 4 (run from the project folder, after analyze.py).
Reads data/raw/vacancies.jsonl and outputs/processed_vacancies.csv; writes outputs/a4/*.csv and figures. Seed 42."""
import re, sys, pathlib, itertools
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, ".")
import analyze as A

SEED = 42
OUT = pathlib.Path("outputs/a4"); OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(SEED)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
BLUE, ORANGE, GREY = "#2b6cb0", "#dd7a1f", "#8a8f98"

proc = pd.read_csv("outputs/processed_vacancies.csv")
proc["published_at"] = pd.to_datetime(proc["published_at"], utc=True)
df = proc[proc.role_group != "Other"].copy()
T = [c for c in df.columns if c.startswith("t_")]
names = [c[2:] for c in T]
n = len(df)

# ---------------------------------------------------------------- E1 weekly counts x role group
wk = df.published_at.dt.tz_convert(None).dt.to_period("W-SUN").apply(lambda p: p.start_time.strftime("%d %b"))
df["week"] = wk
order = sorted(df.week.unique(), key=lambda s: pd.to_datetime(s + " 2026", format="%d %b %Y"))
ct = pd.crosstab(df.week, df.role_group).reindex(order)
ct.to_csv(OUT / "weekly_by_role.csv")
ct.plot(kind="bar", stacked=True, figsize=(6.4, 3.2), colormap="tab20c", width=0.8)
plt.ylabel("vacancies"); plt.xlabel("week starting (Monday)"); plt.xticks(rotation=0)
plt.legend(title="role group", fontsize=7, title_fontsize=7, frameon=False, ncol=2); plt.tight_layout()
plt.savefig(OUT / "fig_weekly_roles.png"); plt.close()

# level x role, city
pd.crosstab(df.role_group, df.level).to_csv(OUT / "role_by_level.csv")
df.area.value_counts().head(8).to_csv(OUT / "city_counts.csv")
tech_per = df[T].sum(axis=1)
tp = df.assign(k=tech_per).groupby("role_group").k.agg(["count", "median", "mean"]).round(2)
tp.to_csv(OUT / "techs_per_vacancy_by_role.csv")
tech_overall = tech_per.describe()

# ---------------------------------------------------------------- E2 salary
s = df[df.salary_currency.eq("KZT") & df.salary_kzt_mid.notna()].copy()
sal_lvl = s.groupby("level").salary_kzt_mid.agg(n="count", median="median", q1=lambda x: x.quantile(.25), q3=lambda x: x.quantile(.75))
sal_lvl.to_csv(OUT / "salary_by_level.csv")
sal_role = s.groupby("role_group").salary_kzt_mid.agg(n="count", median="median", q1=lambda x: x.quantile(.25), q3=lambda x: x.quantile(.75))
sal_role.to_csv(OUT / "salary_by_role.csv")
kw = stats.kruskal(*[g.salary_kzt_mid.values for _, g in s.groupby("level")])
fig, ax = plt.subplots(figsize=(5.2, 3.2))
lv = ["junior", "middle", "senior"]
data = [s[s.level == l].salary_kzt_mid / 1000 for l in lv]
ax.boxplot(data, tick_labels=[f"{l}\n(n={len(x)})" for l, x in zip(lv, data)], showfliers=True, widths=.5,
           medianprops=dict(color=ORANGE), boxprops=dict(color=BLUE), whiskerprops=dict(color=BLUE), capprops=dict(color=BLUE),
           flierprops=dict(markeredgecolor=GREY, markersize=3))
ax.set_yscale("log"); ax.set_ylabel("monthly salary, thousand KZT (log scale)"); plt.tight_layout()
plt.savefig(OUT / "fig_salary_level.png"); plt.close()

# ---------------------------------------------------------------- E3 co-occurrence among the 15 most frequent technologies
top = df[T].sum().sort_values(ascending=False).head(15).index.tolist()
M = df[top].astype(int)
inter = M.T.dot(M).values.astype(float); cnt = np.diag(inter)
union = cnt[:, None] + cnt[None, :] - inter
J = inter / union
pairs = []
for i, j in itertools.combinations(range(len(top)), 2):
    lift = (inter[i, j] / n) / ((cnt[i] / n) * (cnt[j] / n))
    pairs.append((top[i][2:], top[j][2:], int(inter[i, j]), inter[i, j] / n * 100, J[i, j], lift))
pairs = pd.DataFrame(pairs, columns=["tech_a", "tech_b", "both", "both_pct", "jaccard", "lift"])
pairs.sort_values("jaccard", ascending=False).to_csv(OUT / "cooccurrence_pairs.csv", index=False)
fig, ax = plt.subplots(figsize=(5.6, 4.6))
Jp = J.copy(); np.fill_diagonal(Jp, np.nan)
im = ax.imshow(Jp, cmap="Blues", vmin=0, vmax=np.nanmax(Jp))
lab = [t[2:] for t in top]
ax.set_xticks(range(15)); ax.set_xticklabels(lab, rotation=60, ha="right"); ax.set_yticks(range(15)); ax.set_yticklabels(lab)
for i in range(15):
    for j in range(15):
        if i != j and Jp[i, j] >= .25: ax.text(j, i, f"{Jp[i,j]:.2f}", ha="center", va="center", fontsize=6, color="white" if Jp[i,j] > .45 else "black")
plt.colorbar(im, fraction=.046, label="Jaccard index"); plt.tight_layout(); plt.savefig(OUT / "fig_cooccurrence.png"); plt.close()

# ---------------------------------------------------------------- B1 effect sizes with CI (difference in proportions, Newcombe-Wilson)
def wil(k, m, z=1.96):
    p = k / m; d = 1 + z*z/m; c = p + z*z/(2*m); a = z*np.sqrt(p*(1-p)/m + z*z/(4*m*m)); return (c-a)/d, (c+a)/d
def diff_ci(k1, n1, k2, n2):
    p1, p2 = k1/n1, k2/n2; l1, u1 = wil(k1, n1); l2, u2 = wil(k2, n2)
    d = p1 - p2
    return d, d - np.sqrt((p1-l1)**2 + (u2-p2)**2), d + np.sqrt((u1-p1)**2 + (p2-l2)**2)
rows = []
def add(label, mask1, mask2, tech):
    c = "t_" + tech; k1, n1 = int(df.loc[mask1, c].sum()), int(mask1.sum()); k2, n2 = int(df.loc[mask2, c].sum()), int(mask2.sum())
    d, lo, hi = diff_ci(k1, n1, k2, n2); rows.append((label, tech, f"{k1}/{n1}", f"{k2}/{n2}", 100*d, 100*lo, 100*hi))
r = df.role_group; l = df.level
add("Data/ML vs Backend", r == "Data/ML", r == "Backend", "Python")
add("Backend vs Data/ML", r == "Backend", r == "Data/ML", "Java")
add("QA vs Backend", r == "QA", r == "Backend", "CI/CD")
add("QA vs Backend", r == "QA", r == "Backend", "Java")
add("Backend vs QA", r == "Backend", r == "QA", "PostgreSQL")
add("senior vs junior", l == "senior", l == "junior", "Kubernetes")
add("senior vs junior", l == "senior", l == "junior", "Kafka")
add("senior vs junior", l == "senior", l == "junior", "CI/CD")
add("senior vs junior", l == "senior", l == "junior", "Docker")
add("middle vs junior", l == "middle", l == "junior", "Kubernetes")
add("middle vs junior", l == "middle", l == "junior", "CI/CD")
pd.DataFrame(rows, columns=["comparison", "technology", "group1", "group2", "diff_pp", "ci_lo", "ci_hi"]).to_csv(OUT / "effect_sizes_pp.csv", index=False)

# ---------------------------------------------------------------- B2 salary: adjusted association (OLS on log salary, bootstrap CI)
s = s.copy()
s["role4"] = s.role_group.where(s.role_group.isin(["Backend", "Data/ML", "QA"]), "other")
def design(d, tech):
    X = pd.DataFrame({"const": 1.0, "tech": d["t_" + tech].astype(float),
                      "middle": (d.level == "middle").astype(float), "senior": (d.level == "senior").astype(float),
                      "ml": (d.role4 == "Data/ML").astype(float), "qa": (d.role4 == "QA").astype(float), "oth": (d.role4 == "other").astype(float)})
    return X.values, np.log(d.salary_kzt_mid.values)
def coef(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]
adj = []
for tech in ["SQL", "Docker", "PostgreSQL", "Git", "CI/CD", "Python", "Kubernetes", "Kafka", "REST API", "Java"]:
    X, y = design(s, tech)
    if X[:, 1].sum() < 10 or (1 - X[:, 1]).sum() < 10: continue
    b = coef(X, y)[1]
    unadj = np.log(s.loc[s["t_" + tech], "salary_kzt_mid"].median() / s.loc[~s["t_" + tech], "salary_kzt_mid"].median())
    bs = []
    idx = np.arange(len(s))
    for _ in range(2000):
        ii = rng.choice(idx, len(idx)); Xb, yb = X[ii], y[ii]
        if np.linalg.matrix_rank(Xb) < Xb.shape[1]: continue
        bs.append(coef(Xb, yb)[1])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    adj.append((tech, int(X[:, 1].sum()), 100 * (np.exp(unadj) - 1), 100 * (np.exp(b) - 1), 100 * (np.exp(lo) - 1), 100 * (np.exp(hi) - 1)))
pd.DataFrame(adj, columns=["technology", "n_with", "median_ratio_pct", "adj_pct", "ci_lo", "ci_hi"]).to_csv(OUT / "salary_adjusted.csv", index=False)

# ---------------------------------------------------------------- R1..R5 robustness
share = df[T].mean() * 100; share.index = names
def rank_corr(a, b):
    both = pd.concat([a, b], axis=1).dropna(); return stats.spearmanr(both.iloc[:, 0], both.iloc[:, 1])[0]
def overlap(a, b, k=10): return len(set(a.sort_values(ascending=False).head(k).index) & set(b.sort_values(ascending=False).head(k).index))
rob = []
# R1 all IT-category
all_df = A.label(A.clean(A.load())); allit = all_df[T].mean() * 100; allit.index = names
rob.append((f"R1 all IT-category postings (n={len(all_df)})", rank_corr(share, allit), overlap(share, allit), len(all_df)))
# R2 key skills only: apply patterns to key_skills field only; use vacancies with non-empty key skills
ks = df[df.key_skills.map(lambda x: isinstance(x, str) and x not in ("[]", ""))].copy()
import ast
def parse(x):
    try: return [str(i).lower() for i in ast.literal_eval(x)]
    except Exception: return []
ks["sk"] = ks.key_skills.map(parse)
ks_share = {}
for tech, (pat, _) in A.TECH.items():
    rx = re.compile(pat); ks_share[tech] = 100 * ks.sk.map(lambda s_: any(rx.search(x) for x in s_)).mean()
ks_share = pd.Series(ks_share)
ft_same = df.loc[ks.index, T].mean() * 100; ft_same.index = names
rob.append((f"R2 key skills only, vacancies with skills (n={len(ks)})", rank_corr(share, ks_share), overlap(share, ks_share), len(ks)))
rob.append((f"R2b full text, same vacancies (n={len(ks)})", rank_corr(share, ft_same), overlap(share, ft_same), len(ks)))
# R3 temporal split half (by median publication date)
med = df.published_at.median()
h1, h2 = df[df.published_at <= med], df[df.published_at > med]
s1 = h1[T].mean() * 100; s2 = h2[T].mean() * 100; s1.index = s2.index = names
rob.append((f"R3 first half (<= {med:%d %b}, n={len(h1)}) vs second half (n={len(h2)})", rank_corr(s1, s2), overlap(s1, s2), len(df)))
# R4 reposts kept
raw = A.load(); raw["published_at"] = pd.to_datetime(raw["published_at"], utc=True)
raw["desc_text"] = raw["description"].map(A.strip_html)
hit = raw["name"].fillna("").str.lower().str.contains(A.EXCLUDE_TITLE, regex=True)
rp = A.label(raw[~hit & (raw.desc_text.str.len() > 0)].copy()); rp = rp[rp.role_group != "Other"]
rps = rp[T].mean() * 100; rps.index = names
rob.append((f"R4 reposts kept (n={len(rp)})", rank_corr(share, rps), overlap(share, rps), len(rp)))
# R5 bootstrap rank stability
B = 2000; ranks = np.zeros((B, len(T)))
Mx = df[T].values.astype(float)
for b in range(B):
    ii = rng.integers(0, n, n); m = Mx[ii].mean(0)
    ranks[b] = stats.rankdata(-m, method="min")
rs = pd.DataFrame({"technology": names, "observed_rank": stats.rankdata(-Mx.mean(0), method="min"),
                   "rank_lo": np.percentile(ranks, 2.5, axis=0), "rank_hi": np.percentile(ranks, 97.5, axis=0),
                   "p_top3": (ranks <= 3).mean(0), "p_top5": (ranks <= 5).mean(0), "p_top10": (ranks <= 10).mean(0)}).sort_values("observed_rank")
rs.to_csv(OUT / "bootstrap_ranks.csv", index=False)
pd.DataFrame(rob, columns=["check", "spearman_rho", "top10_overlap", "n"]).to_csv(OUT / "robustness.csv", index=False)
# share tables for appendix / figure
cmp_ = pd.DataFrame({"main": share, "all_IT": allit, "key_skills_only": ks_share, "first_half": s1, "second_half": s2, "reposts_kept": rps}).round(1)
cmp_.sort_values("main", ascending=False).to_csv(OUT / "robustness_shares.csv")
# figure: bootstrap rank intervals for top 12
top12 = rs.head(12).iloc[::-1]
fig, ax = plt.subplots(figsize=(5.6, 3.6))
ax.hlines(range(len(top12)), top12.rank_lo, top12.rank_hi, color=BLUE, lw=3)
ax.plot(top12.observed_rank, range(len(top12)), "o", color=ORANGE, ms=5)
ax.set_yticks(range(len(top12))); ax.set_yticklabels(top12.technology); ax.set_xlabel("rank (1 = most frequent); bar = 95% bootstrap interval, dot = observed")
ax.invert_xaxis() if False else None; plt.tight_layout(); plt.savefig(OUT / "fig_rank_stability.png"); plt.close()
# figure: scatter main vs all IT & vs key skills
fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.0))
for ax, ser, ttl in ((axs[0], allit, "all IT-category (n=999)"), (axs[1], ks_share, f"key skills only (n={len(ks)})")):
    ax.scatter(share, ser.reindex(share.index), s=14, color=BLUE); mx = max(share.max(), ser.max()) + 3
    ax.plot([0, mx], [0, mx], color=GREY, lw=.8, ls="--"); ax.set_xlabel("share, main analysis (%)"); ax.set_ylabel(f"share, {ttl} (%)", fontsize=8)
    for t in ("SQL", "Git", "Python", "1C"):
        ax.annotate(t, (share[t], ser[t]), fontsize=6, xytext=(2, 2), textcoords="offset points")
plt.tight_layout(); plt.savefig(OUT / "fig_robustness_scatter.png"); plt.close()

# print summary
print("tech per vacancy:", tech_overall.round(2).to_dict())
print(tp); print(ct); print(sal_lvl.round(0)); print("kruskal", kw)
print(pd.read_csv(OUT/"effect_sizes_pp.csv").round(1).to_string()); print(pd.read_csv(OUT/"salary_adjusted.csv").round(1).to_string())
print(pairs.sort_values("jaccard", ascending=False).head(10).round(2).to_string())
print(pd.read_csv(OUT/"robustness.csv").round(3).to_string()); print(rs.head(10).round(3).to_string())
