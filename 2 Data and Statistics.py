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


# Fit Newton's Law of Cooling for each experimental condition

results = {}
fit_parameters = {}

for j, label in enumerate(labels):

    # Select only valid observations
    valid = ~np.isnan(columns[:, j])
    ts = time[valid]
    T = columns[valid, j]

    # Initial parameter estimates
    Ta0 = room_temps[j] if not np.isnan(room_temps[j]) else T.min()
    T00 = T[0]

    span = ts[-1] - ts[0]

    drop_end = max(T[-1] - Ta0, 1e-6)
    drop_start = max(T00 - Ta0, 1e-6)

    k0 = np.log(drop_start / drop_end) / span if span > 0 else 1.0
    k0 = max(k0, 1e-6)

    # Curve fitting
    opt, cov = curve_fit(
        newton_cooling,
        ts,
        T,
        p0=[Ta0, T00, k0],
        maxfev=10000
    )

    unc = np.sqrt(np.diag(cov))

    Ta_opt, T0_opt, k_opt = opt
    Ta_unc, T0_unc, k_unc = unc

    # Calculate fitted temperatures and residuals
    fit = newton_cooling(ts, *opt)
    res = T - fit

    rmse = np.sqrt(np.mean(res**2))

    # Store observations and fitted curves
    results[label] = (ts, T, fit, res)

    # Store fitted parameters and their uncertainties
    fit_parameters[label] = {
        "Ta": Ta_opt,
        "T0": T0_opt,
        "k": k_opt,
        "Ta_unc": Ta_unc,
        "T0_unc": T0_unc,
        "k_unc": k_unc,
        "RMSE": rmse
    }

    print(f"{label} ({len(ts)} points)")
    print(f"Ambient Temp (Ta): {Ta_opt:.2f} ± {Ta_unc:.2f} °C")
    print(f"Initial Temp (T0): {T0_opt:.2f} ± {T0_unc:.2f} °C")
    print(f"Cooling Const (k): {k_opt:.5f} ± {k_unc:.5f} min⁻¹")
    print(f"RMSE: {rmse:.3f} °C\n")


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



# ============================================================
# 2.2 STATISTICAL COMPARISON OF COOLING CONSTANTS
# Fan vs No Fan AND With Lid vs Without Lid
# ============================================================

import numpy as np
from scipy import stats


def compare_cooling_constants(label1, label2, k1, k2, se1, se2,
                               n1, n2, increase_description):

    # Difference in cooling constants
    delta_k = k1 - k2

    # Standard error of the difference
    se_diff = np.sqrt(se1**2 + se2**2)

    # Welch-Satterthwaite approximate degrees of freedom
    df = (
        (se1**2 + se2**2)**2 /
        (
            se1**4 / (n1 - 3) +
            se2**4 / (n2 - 3)
        )
    )

    # t-statistic and two-tailed p-value
    t_stat = delta_k / se_diff
    p_value = 2 * stats.t.sf(abs(t_stat), df)

    # 95% confidence interval
    t_critical = stats.t.ppf(0.975, df)
    ci_lower = delta_k - t_critical * se_diff
    ci_upper = delta_k + t_critical * se_diff

    # Percentage increase
    percentage_increase = (delta_k / k2) * 100

    print(f"\n{'='*55}")
    print(f"{label1} vs {label2}")
    print(f"{'='*55}")

    print(f"Difference in cooling constants: {delta_k:.6f} min⁻¹")
    print(f"Standard error of difference: {se_diff:.6f}")
    print(f"Degrees of freedom: {df:.2f}")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.6e}")
    print(f"95% CI for difference: [{ci_lower:.6f}, {ci_upper:.6f}] min⁻¹")
    print(f"Percentage {increase_description}: {percentage_increase:.2f}%")

    if p_value < 0.05:
        print("\nConclusion: Reject the null hypothesis.")
        print("There is a statistically significant difference")
        print("between the cooling constants.")
    else:
        print("\nConclusion: Fail to reject the null hypothesis.")
        print("There is no statistically significant difference.")

    return {
        "delta_k": delta_k,
        "se_diff": se_diff,
        "df": df,
        "t_stat": t_stat,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "percentage_increase": percentage_increase
    }


# ============================================================
# A. FAN vs NO FAN
# ============================================================

fan_label = labels[1]
no_fan_label = labels[0]

fan_results = compare_cooling_constants(
    label1=fan_label,
    label2=no_fan_label,
    k1=fit_parameters[fan_label]["k"],
    k2=fit_parameters[no_fan_label]["k"],
    se1=fit_parameters[fan_label]["k_unc"],
    se2=fit_parameters[no_fan_label]["k_unc"],
    n1=len(results[fan_label][0]),
    n2=len(results[no_fan_label][0]),
    increase_description=f"increase with {fan_label}"
)


# ============================================================
# B. WITHOUT LID vs WITH LID
# ============================================================

without_lid_label = labels[3]
with_lid_label = labels[4]

lid_results = compare_cooling_constants(
    label1=without_lid_label,
    label2=with_lid_label,
    k1=fit_parameters[without_lid_label]["k"],
    se1=fit_parameters[without_lid_label]["k_unc"],
    k2=fit_parameters[with_lid_label]["k"],
    se2=fit_parameters[with_lid_label]["k_unc"],
    n1=len(results[without_lid_label][0]),
    n2=len([with_lid_label][0]),
    increase_description=f"increase with {without_lid_label}"
)

