import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binned_statistic

def uneven_allan_deviation(
    timestamps, values, taus, overlap_factor=1, min_fill_ratio=0.5, min_abs_samples=1
    ):
  """
  Computes Allan Deviation for unevenly sampled time series.
  Set overlap_factor=1 for standard non-overlapping, or >1 (e.g., 4 or 8) for overlapping.
  Use adaptive median-count threshold to reject partially empty gap bins.
  """
  t = np.asarray(timestamps, dtype=float)
  y = np.asarray(values, dtype=float)
  t_start, t_end = t.min(), t.max()

  valid_taus = []
  adevs = []

  for tau in taus:
    step = tau / overlap_factor
    shift_means = []
    shift_counts = []

		# 1. Collect bin means and counts across all K shifts first
    for k in range(overlap_factor):
      shift = k * step
      bins = np.arange(t_start + shift, t_end + tau, tau)
      if len(bins) < 3:
        continue

      means, _, _ = binned_statistic(t, y, statistic="mean", bins=bins)
      counts, _, _ = binned_statistic(t, y, statistic="count", bins=bins)
      
      shift_means.append(means)
      shift_counts.append(counts)
	   
	  if not shift_counts:
      continue

		# 2. Find median count among non-empty bins across all shifts
		all_counts = np.concatenate(shift_counts)
    non_empty = all_counts[all_counts > 0]
    if len(non_empty) == 0:
      continue
      
    median_count = np.median(non_empty)
    threshold = max(min_abs_samples, min_fill_ratio * median_count)
    
    # 3. Filter each shift using the adaptive threshold and pool adjacent diffs
    all_valid_diffs = []
    for means, counts in zip(shift_means, shift_counts):
       filtered_means = np.where(counts >= threshold, means, np.nan)
       diffs = filtered_means[1:] - filtered_means[:-1]
       valid_diffs = diffs[~np.isnan(diffs)]
       if len(valid_diffs) > 0:
         all_valid_diffs.append(valid_diffs)

    if not all_valid_diffs:
      continue
    
    pooled_diffs = np.concatenate(all_valid_diffs)

    if len(pooled_diffs) >= 2:
      avar = 0.5 * np.mean(pooled_diffs**2)
      valid_taus.append(tau)
      adevs.append(np.sqrt(avar))

  return np.array(valid_taus), np.array(adevs)

# --- Simulation & Plotting ---
if __name__ == "__main__":
  rng = np.random.default_rng(42)

  # 1. Generate 15,000 unevenly sampled timestamps (~3,000 seconds total)
  dt = rng.exponential(scale=0.2, size=15000)
  timestamps = np.cumsum(dt)

  # 2. Generate signal: high-frequency white noise + slow random walk drift
  white_noise = rng.normal(0, 1.0, size=15000)
  random_walk = np.cumsum(rng.normal(0, 0.025, size=15000))
  values = white_noise + random_walk

  # 3. Define logarithmically spaced averaging times (taus)
  min_tau = np.percentile(dt, 90) * 2
  max_tau = (timestamps[-1] - timestamps[0]) / 4.0  # <-- Fixed: timestamps[0]
  taus = np.logspace(np.log10(min_tau), np.log10(max_tau), 35)

  # 4. Compute Non-Overlapping (overlap_factor=1) and Overlapping (overlap_factor=8)
  taus_non, adev_non = uneven_allan_deviation(
      timestamps, values, taus, overlap_factor=1, min_samples_per_bin=2
  )
  taus_ovl, adev_ovl = uneven_allan_deviation(
      timestamps, values, taus, overlap_factor=8, min_samples_per_bin=2
  )

  # 5. Plot comparison on a log-log scale
  plt.figure(figsize=(9, 5.5))
  plt.loglog(
      taus_non,
      adev_non,
      "o--",
      alpha=0.75,
      markersize=5,
      label="Non-Overlapping ADEV (overlap=1)",
  )
  plt.loglog(
      taus_ovl,
      adev_ovl,
      "-",
      linewidth=2.5,
      label="Overlapping ADEV (overlap=8)",
  )

  # Add reference slopes (-1/2 for white noise, +1/2 for random walk)
  ref_tau_short = taus_ovl[:8]
  plt.loglog(
      ref_tau_short,
      adev_ovl[0] * (ref_tau_short / ref_tau_short[0]) ** (-0.5),  # <-- Fixed
      ":",
      color="gray",
      label="Slope -1/2 (White Noise)",
  )

  ref_tau_long = taus_ovl[-10:]
  plt.loglog(
      ref_tau_long,
      adev_ovl[-1] * (ref_tau_long / ref_tau_long[-1]) ** (0.5),
      "-.",
      color="black",
      label="Slope +1/2 (Random Walk)",
  )

  plt.xlabel("Averaging Time $\\tau$ (s)", fontsize=11)
  plt.ylabel("Allan Deviation $\\sigma(\\tau)$", fontsize=11)
  plt.title(
      "Unevenly Sampled Data: Non-Overlapping vs. Overlapping Allan Deviation",
      fontsize=12,
  )
  plt.grid(True, which="both", linestyle="--", alpha=0.5)
  plt.legend()
  plt.tight_layout()
  plt.show()