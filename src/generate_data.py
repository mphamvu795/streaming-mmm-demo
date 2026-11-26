"""Generate synthetic weekly marketing data for a fictional streaming service ("Lumen+").

Because the data is simulated, the true effect of every channel is known. That lets us
check whether the MMM recovers the truth - something you can never do with real data.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
OUT = Path(__file__).resolve().parents[1] / "data"
OUT.mkdir(exist_ok=True)

N_WEEKS = 156  # three years
dates = pd.date_range("2023-01-02", periods=N_WEEKS, freq="W-MON")
t = np.arange(N_WEEKS)

# --- True channel behaviour (hidden from the model) ---------------------------------
# alpha: adstock decay (how long an ad keeps working). lam: how quickly the channel saturates.
# beta: max weekly sign-ups the channel can drive once fully saturated.
CHANNELS = {
    #            mean weekly spend ($)  alpha  lam   beta (sign-ups)
    "meta":     dict(spend=180_000, alpha=0.40, lam=1.6, beta=4200),
    "tiktok":   dict(spend=90_000,  alpha=0.20, lam=1.2, beta=3000),
    "youtube":  dict(spend=110_000, alpha=0.60, lam=2.5, beta=1600),
    "search":   dict(spend=60_000,  alpha=0.10, lam=3.0, beta=2000),
}


def geometric_adstock(x, alpha, l_max=8):
    w = alpha ** np.arange(l_max)
    w = w / w.sum()
    return np.convolve(x, w)[: len(x)]


def logistic_saturation(x, lam):
    return (1 - np.exp(-lam * x)) / (1 + np.exp(-lam * x))


df = pd.DataFrame({"date": dates})

# Spend: seasonal flighting + noise + occasional bursts and dark weeks.
for ch, p in CHANNELS.items():
    season = 1 + 0.25 * np.sin(2 * np.pi * (t + RNG.integers(0, 52)) / 52)
    noise = RNG.lognormal(0, 0.25, N_WEEKS)
    burst = np.where(RNG.random(N_WEEKS) < 0.08, RNG.uniform(1.5, 2.2, N_WEEKS), 1)
    dark = np.where(RNG.random(N_WEEKS) < 0.05, 0, 1)
    df[f"{ch}_spend"] = np.round(p["spend"] * season * noise * burst * dark, -2)

# Content launches: ~10 major original releases a year, effect lasts ~3 weeks.
launch = (RNG.random(N_WEEKS) < 0.19).astype(int)
launch_effect = np.convolve(launch, [1.0, 0.55, 0.25])[:N_WEEKS]
df["content_launch"] = launch

# Holiday period (late Dec / early Jan) uplift.
week_of_year = dates.isocalendar().week.to_numpy()
df["holiday"] = ((week_of_year >= 51) | (week_of_year <= 1)).astype(int)

# --- Build sign-ups from the true process ----------------------------------------
baseline = 9000 + 18 * t  # slow organic growth
seasonality = 900 * np.sin(2 * np.pi * t / 52 + 1.2) + 400 * np.cos(4 * np.pi * t / 52)
contrib = {}
for ch, p in CHANNELS.items():
    x = df[f"{ch}_spend"].to_numpy()
    x_scaled = x / x.max()
    contrib[ch] = p["beta"] * logistic_saturation(geometric_adstock(x_scaled, p["alpha"]), p["lam"])

signups = (
    baseline
    + seasonality
    + 3500 * launch_effect
    + 1800 * df["holiday"].to_numpy()
    + sum(contrib.values())
    + RNG.normal(0, 600, N_WEEKS)
)
df["new_subscribers"] = np.round(signups).astype(int)
df.to_csv(OUT / "lumen_weekly.csv", index=False)

# Save the ground truth so we can grade the model later.
truth = {
    ch: {
        "alpha": p["alpha"],
        "total_signups": float(contrib[ch].sum()),
        "total_spend": float(df[f"{ch}_spend"].sum()),
        "cost_per_signup": float(df[f"{ch}_spend"].sum() / contrib[ch].sum()),
    }
    for ch, p in CHANNELS.items()
}
(OUT / "ground_truth.json").write_text(json.dumps(truth, indent=2))
print(df.describe().round(0).T[["mean", "min", "max"]])
print(json.dumps(truth, indent=2))
