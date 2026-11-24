# Stride

Stride turns workout recordings from many devices into one clean, versioned dataset.

A watch writes binary FIT files in UTC and metres. A phone app writes GPX. A bike computer
writes TCX and records less often when riding steadily. A fitness app exports CSV in miles,
mph and feet, with local time and no time zone. Their clocks disagree by seconds, and every
one of them sometimes records a bad GPS fix, a repeated timestamp or a heart rate of 250.
Stride reads all of them, converts them to one schema, finds what is wrong, lines up their
clocks, merges them, and writes a snapshot that can be checked and reproduced.

## Pipeline

```
files (FIT / GPX / TCX / CSV), one folder per device
  -> read          own FIT binary decoder; GPX, TCX, CSV readers
  -> normalize     device units -> SI, local time -> UTC (from the device registry)
  -> validate      8 sample rules + a unit check, per-file quality metrics
  -> clean         sort, drop repeated timestamps, blank only the flagged channel
  -> group         overlapping recordings become one session
  -> sync          estimate each device's clock offset and drift against a reference
  -> resample      every device onto the same whole-second grid, no filling across gaps
  -> fuse          per channel, take the most trusted device that has a value
  -> snapshot      Parquet tables + reports + manifest, id = hash of content and inputs
  -> check         independent release checks on the snapshot as written to disk
```

## Device registry

`src/stride/devices.yaml` says how each device writes its files: format, units, time zone
for timestamps without an offset, and nominal sample rate. The pipeline never guesses
units: if a file disagrees with its registry entry, validation reports it.

```yaml
app_csv:
  format: csv
  timezone: Asia/Kolkata
  nominal_hz: 0.5
  units: {distance: mi, speed: mph, elevation: ft}
  csv_columns: {t: Time, lat: Latitude, ...}
```

## Validation

Rules run on the data in the order the device wrote it, before anything is cleaned.

| Rule | Flags | Cleaning |
|---|---|---|
| `non_monotonic` | a sample earlier than one before it | sorted |
| `duplicate_timestamp` | a repeated time | dropped |
| `timestamp_gap` | first sample after a pause > 5 nominal periods (min 10 s) | kept, marked |
| `gps_jump` | a fix further from the last trusted fix than the sport allows (+20 m noise) | position blanked |
| `impossible_speed` | speed < 0 or above the sport's limit | speed blanked |
| `hr_out_of_range` | heart rate outside 30–230 bpm | heart rate blanked |
| `elevation_spike` | > 40 m from the 11-sample median | elevation blanked |
| `distance_decrease` | cumulative distance going backwards | distance blanked |

GPS jumps are judged against the last trusted fix, not the previous one, so a
three-sample excursion is caught whole. After five jumps in a row the track is accepted as
moved (for example after a car ride), so one bad anchor cannot flag the rest of a file.

**Unit check.** If the registry says unit R but the device wrote unit A, values come out
scaled by factor(R)/factor(A). Stride compares the device's distance and time-weighted
speed with the GPS track (excluding flagged jumps) and names the mix-up, for example
"registered as mi, but the data looks like km". It reports instead of rescaling: the fix
belongs in the registry.

Each file also gets quality metrics: completeness per channel, validity (share of samples
with no error flag) and achieved rate over nominal rate.

## Clock sync

Two devices on the same run see the same path. Stride slides device B's track in time over
the reference track and picks the lag with the smallest mean squared distance: first in
1 s steps over ±300 s, then in 0.1 s steps refined with a parabola.

Two details matter for accuracy:

- **Interpolation bias.** Comparing at a lag that lands between B's samples means
  interpolating, which averages two noisy fixes and lowers the cost. Uncorrected, this
  pulled estimates about 0.14 s towards half-sample lags. Stride estimates B's white noise
  from second differences (MAD-based) and subtracts the expected term.
- **Drift needs evidence.** GPS error is correlated over tens of seconds, so lags measured
  in 10-minute windows wobble by a few tenths of a second. Drift is fitted across windows
  and kept only if the slope is at least three standard errors from zero.

## Snapshots

```
out/snapshots/<id>/
  recordings.parquet   every device, cleaned, synced, resampled (canonical schema)
  sessions.parquet     one fused row per second per session
  reports.json         validation, sync and provenance per file and session
  manifest.json        schema and pipeline version, config, input SHA-256s, content hashes
```

The id hashes the schema version, pipeline version, config, input file hashes and table
content (hashed from values, not Parquet bytes). Same inputs and settings give the same id.
`stride check` re-reads a snapshot from disk and checks content hashes, schema, time order,
the resample grid, value ranges and a quality gate. `stride diff` compares two snapshots.

## Results

On a seeded synthetic benchmark with exact ground truth: 200 sessions, every fault labelled,
every clock error known. Full tables in [docs/results.md](docs/results.md).

| | Result |
|---|---|
| Data | 200 sessions, 153.8 hours, 425 files (149 watch_fit, 134 phone_gpx, 79 bike_tcx, 63 app_csv), 847,028 samples |
| Fault detection | 1713 injected faults across 8 types: precision 1.000, recall 0.962 |
| Unit mix-ups | 13 of 13 mislabelled files caught, 0 false alarms |
| Clock sync | 225 device pairs: median offset error 0.09 s, 99.1% within 0.5 s |
| Distance | median error 0.35% (naive sum of raw fixes: 6.70%) |
| Throughput | 16,280 samples/s on one core (arm64 Darwin, Python 3.12.13) |
| Reproducibility | two runs give the same snapshot id: yes |

The benchmark data is synthetic because it needs exact ground truth: real files do not come
with labels for every bad sample or the true clock error. The generator writes real FIT,
GPX, TCX and CSV files, so the readers are exercised exactly as with device exports.

## Usage

```bash
pip install -e ".[dev]"

stride ingest path/to/inputs --out out      # one subfolder per device id in the registry
stride check out/snapshots/<id>
stride show out/snapshots/<id>
stride diff out/snapshots/<old> out/snapshots/<new>

stride synth --seed 7 --sessions 20 --out data/synth
stride bench --seed 7 --sessions 200 --markdown docs/results.md
```

## Layout

```
src/stride/
  schema.py registry.py devices.yaml units.py    data model and device registry
  io/                                            FIT decoder/encoder, GPX, TCX, CSV readers
  normalize.py geo.py resample.py                units, time zones, ENU frame, resampling
  validation/ clean.py                           rules, unit check, quality metrics, cleaning
  sync.py fuse.py pipeline.py                    clock sync, fusion, orchestration
  snapshot.py check.py diff.py cli.py            versioned output and release checks
  synth/                                         ground truth, device simulation, faults, benchmark
```

## Limits

- The FIT decoder reads the messages Stride needs (record, sport, session); others are
  walked and skipped. Developer fields are skipped, not decoded.
- Clock sync needs GPS on both devices. A heart-rate-only device is not synced.
- Fusion picks one device per channel per second; it does not average positions.
