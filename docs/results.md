# Benchmark results

Reproduce with `stride bench --seed 7 --sessions 200`. The dataset is synthetic: ground truth is known exactly,
so every number below is checked against it. See the README for how it is generated.

## Dataset

- 200 sessions, 153.8 hours, 425 files, 847,028 samples
- Files by device: watch_fit 149, phone_gpx 134, bike_tcx 79, app_csv 63
- Files that failed to parse: 0

## Fault detection (sample level)

Files written in the wrong units are scored in the unit-check section instead.

| Rule | Injected | Found | False alarms | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| gps_jump | 777 | 740 | 0 | 1.000 | 0.952 |
| elevation_spike | 166 | 149 | 0 | 1.000 | 0.898 |
| hr_out_of_range | 133 | 133 | 0 | 1.000 | 1.000 |
| impossible_speed | 120 | 113 | 0 | 1.000 | 0.942 |
| distance_decrease | 54 | 54 | 0 | 1.000 | 1.000 |
| duplicate_timestamp | 145 | 145 | 0 | 1.000 | 1.000 |
| non_monotonic | 194 | 194 | 0 | 1.000 | 1.000 |
| timestamp_gap | 124 | 120 | 0 | 1.000 | 0.968 |
| **all** | 1713 | 1648 | 0 | 1.000 | 0.962 |

Missed faults are mostly ones injected below a rule's threshold on purpose (for
example a 40 m GPS jump on a bike computer that records every 8 s).

## Unit check (file level)

- Files written in different units than their registry entry: 13
- Flagged: 13, missed: 0, false alarms: 0
- Precision 1.000, recall 1.000

## Clock sync

- Device pairs synced: 225 (failed: 0)
- Offset error: median 0.091 s, p95 0.3501 s, max 0.6306 s
- Within 0.25 s: 87.1%; within 0.5 s: 99.1%
- Drift error: median 25.0302 ppm, p95 147.9045 ppm

## Distance

Absolute error of session distance against the true distance over the same span.

| Method | Median | Mean | p95 |
|---|---:|---:|---:|
| Pipeline (cleaned, fused, 5 s smoothing) | 0.35% | 0.81% | 3.04% |
| Naive (raw fixes summed, no cleaning) | 6.70% | 16.71% | 56.42% |

## Performance and reproducibility

- Pipeline time: 52.03 s (8.2 files/s, 16,280 samples/s) on one core (arm64 Darwin, Python 3.12.13)
- Two runs gave the same snapshot id: yes
