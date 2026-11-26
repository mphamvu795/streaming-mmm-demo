"""Fit a Bayesian marketing mix model to the Lumen+ data with PyMC-Marketing."""
import os
import warnings
from pathlib import Path

import pandas as pd
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
CHANNELS = ["meta_spend", "tiktok_spend", "youtube_spend", "search_spend"]
CONTROLS = ["content_launch", "holiday", "trend"]
QUICK = os.environ.get("QUICK") == "1"

df = pd.read_csv(ROOT / "data" / "lumen_weekly.csv", parse_dates=["date"])
df["trend"] = range(len(df))  # captures slow organic subscriber growth
X = df[["date", *CHANNELS, *CONTROLS]]
y = df["new_subscribers"]

mmm = MMM(
    date_column="date",
    channel_columns=CHANNELS,
    target_column="new_subscribers",
    control_columns=CONTROLS,
    adstock=GeometricAdstock(l_max=8),   # ads keep working for up to 8 weeks
    saturation=LogisticSaturation(),     # diminishing returns on spend
    yearly_seasonality=2,                # Fourier terms for annual seasonality
)

mmm.fit(
    X, y,
    draws=200 if QUICK else 1000,
    tune=200 if QUICK else 1000,
    chains=2 if QUICK else 4,
    nuts_sampler="numpyro",
    target_accept=0.9,
    random_seed=7,
    progressbar=False,
)
mmm.sample_posterior_predictive(X, extend_idata=True, combined=True, progressbar=False)

out = ROOT / "outputs"
out.mkdir(exist_ok=True)
mmm.save(str(out / "lumen_mmm.nc"))
print(list(mmm.idata.posterior.data_vars))
