"""Turn the fitted MMM into charts, a validation check and a budget recommendation."""
import json
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
NAMES = ["Meta", "TikTok", "YouTube", "Search"]
COLS = ["meta_spend", "tiktok_spend", "youtube_spend", "search_spend"]
COLORS = ["#3b6fd8", "#e0457b", "#d9822b", "#2a9d8f"]
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})

df = pd.read_csv(ROOT / "data" / "lumen_weekly.csv", parse_dates=["date"])
truth = json.loads((ROOT / "data" / "ground_truth.json").read_text())
idata = az.from_netcdf(OUT / "lumen_mmm.nc")
post = az.extract(idata, group="posterior")  # stacks chain/draw into "sample"

# Model works in scaled units; recover the target scale to convert back to sign-ups.
scaled_total = post["channel_contribution"].sum(("date", "channel"))
y_scale = float((post["total_media_contribution_original_scale"] / scaled_total).mean())
contrib = post["channel_contribution"] * y_scale  # sign-ups, dims: date, channel, sample

# --- 1. Model fit ---------------------------------------------------------------------
mu = post["mu"] * y_scale
fig, ax = plt.subplots(figsize=(10, 3.6))
ax.plot(df["date"], df["new_subscribers"], color="#222", lw=1.2, label="Actual")
ax.plot(df["date"], mu.mean("sample"), color=COLORS[0], lw=1.5, label="Model")
ax.fill_between(df["date"], *np.percentile(mu, [3, 97], axis=-1), color=COLORS[0], alpha=0.2)
r2 = 1 - np.sum((df["new_subscribers"] - mu.mean("sample")) ** 2) / np.sum(
    (df["new_subscribers"] - df["new_subscribers"].mean()) ** 2)
mape = float(np.mean(np.abs(df["new_subscribers"] - mu.mean("sample")) / df["new_subscribers"]))
ax.set_title(f"Weekly new subscribers: actual vs model  (R² = {r2:.2f}, MAPE = {mape:.1%})", loc="left")
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT / "1_model_fit.png"); plt.close(fig)

# --- 2. What drove sign-ups ---------------------------------------------------------
ctrl = post["control_contribution"].mean("sample") * y_scale
parts = {
    "Baseline & trend": (post["intercept_contribution"].mean() * y_scale
                         + ctrl.sel(control="trend")).values,
    "Seasonality & holidays": (post["yearly_seasonality_contribution"].mean("sample") * y_scale
                               + ctrl.sel(control="holiday")).values,
    "Content launches": ctrl.sel(control="content_launch").values,
}
for n, c in zip(NAMES, COLS):
    parts[n] = contrib.sel(channel=c).mean("sample").values
fig, ax = plt.subplots(figsize=(10, 4))
ax.stackplot(df["date"], *parts.values(), labels=parts.keys(),
             colors=["#c9ced6", "#9aa3b0", "#6c757d", *COLORS], alpha=0.9)
ax.set_title("What drove weekly sign-ups", loc="left")
ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "2_contributions.png"); plt.close(fig)

# --- 3. Cost per sign-up vs the known truth ----------------------------------------
cps = {}
for n, c in zip(NAMES, COLS):
    draws = df[c].sum() / contrib.sel(channel=c).sum("date").values
    cps[n] = dict(mean=float(np.median(draws)), lo=float(np.percentile(draws, 3)),
                  hi=float(np.percentile(draws, 97)), true=truth[c.split("_")[0]]["cost_per_signup"])
fig, ax = plt.subplots(figsize=(7, 3.6))
yy = np.arange(len(NAMES))
for i, n in enumerate(NAMES):
    ax.errorbar(cps[n]["mean"], i, xerr=[[cps[n]["mean"] - cps[n]["lo"]], [cps[n]["hi"] - cps[n]["mean"]]],
                fmt="o", color=COLORS[i], capsize=4, label="Model estimate (94% interval)" if i == 0 else None)
    ax.plot(cps[n]["true"], i, "x", color="#222", ms=9, mew=2, label="True value" if i == 0 else None)
ax.set_yticks(yy, NAMES); ax.invert_yaxis()
ax.set_xscale("log")
ax.set_xticks([25, 50, 100, 250, 500, 1000, 2500], ["$25", "$50", "$100", "$250", "$500", "$1k", "$2.5k"])
ax.set_xlabel("Cost per incremental sign-up (log scale)")
ax.set_title("Did the model recover the truth?", loc="left")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "3_cost_per_signup_vs_truth.png"); plt.close(fig)

# --- 4. Response curves & budget optimisation --------------------------------------
sub = np.random.default_rng(0).choice(post.sizes["sample"], 500, replace=False)
beta = post["saturation_beta"].values[:, sub] * y_scale  # (channel, draws)
lam = post["saturation_lam"].values[:, sub]
xmax = df[COLS].max().values[:, None]


def weekly_signups(spend):
    """Expected steady-state weekly sign-ups per channel for a weekly spend vector."""
    x = lam * (np.asarray(spend)[:, None] / xmax)
    return (beta * (1 - np.exp(-x)) / (1 + np.exp(-x))).mean(axis=1)


current = df[COLS].mean().values
budget = current.sum()
res = minimize(lambda s: -weekly_signups(s).sum(), current, method="SLSQP",
               bounds=[(0.5 * c, 1.5 * c) for c in current],  # keep changes realistic: +/-50%
               constraints={"type": "eq", "fun": lambda s: s.sum() - budget})
optimal = res.x
base_su, opt_su = weekly_signups(current), weekly_signups(optimal)

fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), sharey=True)
for i, ax in enumerate(axes):
    grid = np.linspace(0, 2.2 * current[i], 80)
    curve = [weekly_signups(np.where(np.arange(4) == i, g, 0))[i] for g in grid]
    ax.plot(grid / 1e3, curve, color=COLORS[i], lw=2)
    ax.axvline(current[i] / 1e3, color="#222", ls="--", lw=1)
    ax.axvline(optimal[i] / 1e3, color=COLORS[i], ls=":", lw=1.5)
    ax.set_title(NAMES[i], loc="left"); ax.set_xlabel("Weekly spend ($k)")
axes[0].set_ylabel("Weekly sign-ups")
fig.suptitle("Response curves  (dashed = current spend, dotted = recommended)", x=0.01, ha="left")
fig.tight_layout(); fig.savefig(OUT / "4_response_curves.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 3.4))
w = 0.38
ax.bar(yy - w / 2, current / 1e3, w, color="#c9ced6", label="Current")
ax.bar(yy + w / 2, optimal / 1e3, w, color=COLORS, label="Recommended")
for i in range(4):
    ax.text(i + w / 2, optimal[i] / 1e3, f"{optimal[i] / current[i] - 1:+.0%}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(yy, NAMES); ax.set_ylabel("Avg weekly spend ($k)")
gain = opt_su.sum() / base_su.sum() - 1
ax.set_title(f"Same budget, reallocated: {gain:+.1%} media-driven sign-ups", loc="left")
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT / "5_budget_reallocation.png"); plt.close(fig)

alpha = post["adstock_alpha"].values
results = dict(
    r2=round(float(r2), 3), mape=round(mape, 4), y_scale=y_scale,
    cost_per_signup=cps,
    adstock_alpha={n: dict(est=float(np.median(alpha[i])), true=truth[c.split("_")[0]]["alpha"])
                   for i, (n, c) in enumerate(zip(NAMES, COLS))},
    weekly_budget=float(budget),
    reallocation={n: dict(current=float(current[i]), recommended=float(optimal[i]),
                          change=float(optimal[i] / current[i] - 1)) for i, n in enumerate(NAMES)},
    media_signups_weekly=dict(current=float(base_su.sum()), recommended=float(opt_su.sum()), uplift=float(gain)),
    annual_extra_signups=float((opt_su.sum() - base_su.sum()) * 52),
)
(OUT / "results.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
