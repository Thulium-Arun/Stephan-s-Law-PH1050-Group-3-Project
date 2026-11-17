import csv
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import probplot


def newton_cooling(t, Ta, T0, k):
    return Ta + (T0 - Ta) * np.exp(-k * t)


path = sys.argv[1]

# Line 1: room temperature per column, line 2: column names, rest: data
with open(path, newline="") as f:
    reader = csv.reader(f)
    room_row = next(reader)
    names = next(reader)

room_temps = [float(x) if x.strip() else np.nan for x in room_row[1:]]
labels = names[1:]  # first column is time

# genfromtxt turns blank cells into NaN instead of raising an error
data = np.genfromtxt(path, delimiter=",", skip_header=2)
time = data[:, 0]
columns = data[:, 1:]

results = {}  # label -> (t, T, fit, residuals)

for j, label in enumerate(labels):
    # only use the rows that actually have data for this column
    valid = ~np.isnan(columns[:, j])
    ts = time[valid]
    T = columns[valid, j]

    # starting guesses come from the data itself
    Ta0 = room_temps[j] if not np.isnan(room_temps[j]) else T.min()
    T00 = T[0]
    span = ts[-1] - ts[0]
    drop_end = max(T[-1] - Ta0, 1e-6)
    drop_start = max(T00 - Ta0, 1e-6)
    k0 = np.log(drop_start / drop_end) / span if span > 0 else 1.0
    k0 = max(k0, 1e-6)

    opt, cov = curve_fit(newton_cooling, ts, T, p0=[Ta0, T00, k0], maxfev=10000)
    unc = np.sqrt(np.diag(cov))
    Ta_opt, T0_opt, k_opt = opt
    Ta_unc, T0_unc, k_unc = unc

    fit = newton_cooling(ts, *opt)
    res = T - fit
    rmse = np.sqrt(np.mean(res ** 2))
    results[label] = (ts, T, fit, res)

    print(f"{label}  ({len(ts)} points, t = {ts[0]:g} to {ts[-1]:g} min)")
    print(f"Ambient Temp (Ta):  {Ta_opt:.2f} ± {Ta_unc:.2f} °C  (room temp in file: {room_temps[j]:g})")
    print(f"Initial Temp (T0):  {T0_opt:.2f} ± {T0_unc:.2f} °C")
    print(f"Cooling Const (k):  {k_opt:.5f} ± {k_unc:.5f} min⁻¹")
    print(f"RMSE:               {rmse:.3f} °C\n")

# Residuals vs time
plt.figure()
for label, (ts, T, fit, res) in results.items():
    plt.scatter(ts, res, s=10, label=label)
plt.axhline(0, color="black")
plt.xlabel("Time (min)")
plt.ylabel("Residual (°C)")
plt.legend()
plt.savefig("residual_scatter.png")
plt.close()

# Histograms
plt.figure()
for label, (ts, T, fit, res) in results.items():
    plt.hist(res, alpha=0.5, label=label)
plt.xlabel("Residual (°C)")
plt.legend()
plt.savefig("residual_histogram.png")
plt.close()

# Q-Q plots, one per dataset
for label, (ts, T, fit, res) in results.items():
    plt.figure()
    probplot(res, plot=plt)
    plt.title(f"Q-Q plot for {label}")
    plt.savefig(f"QQ_{label}.png")
    plt.close()
