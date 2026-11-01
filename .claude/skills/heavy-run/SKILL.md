---
name: heavy-run
description: Launch, monitor, diagnose and kill a long credit-game job -- creditsurv select, report, views, windows, fit, family, moratorium, aggregate, ingest, or any script that reads the cell file. Use before starting one, when one seems slow or stuck, and before killing or restarting one.
---

# Heavy runs

A heavy job here is tens of minutes to hours at several GB on a 16 GB laptop. Two hard rules come
first, and the PreToolUse hook enforces both: **one heavy job at a time**, and **never on battery**.

## Before launching

1. **Is the work already cached?** A selection resumes from cached fits; a fit of the same
   specification on the same table is found by `find_fits(...)`; a reading of the table is written
   once and mapped back in 0.4 s. A run that would recompute any of these is a run to rethink.
2. **Does it hold the panel?** Nothing should. Every command that used to expand the training half
   reads the cell file a batch at a time. A script that calls `cells_to_episodes` on a window of
   years, or a path that stores a design (`fit_streamed`), is the path that thrashed:
   - `creditsurv report` through a stored 4.1 GB design plus the autograd tape ran **31 hours** and
     reached its first evaluation, at 202 M swap-ins;
   - a scoring script holding three years of observation reached **7.25 GB after 4 hours**;
   - the same scoring streamed in 250,000-cell batches ran at flat memory.
   - **`creditsurv report` still holds the panel**: `_split` calls `split_cells(load_cells(...))`,
     which loads the whole cell table and expands it half by half, about 12 GB before the
     first fit. Never launch it beside agents or another job. Moving it to the encoding path
     the campaign already uses is the open fix.
3. **Mains power, and nothing else heavy.** `pmset -g batt` and the session-start status line.

## Launching

```
caffeinate -dimsu uv run creditsurv <command> ... > "$SCRATCHPAD/<name>.log" 2>&1
```

Run it as a background Bash task and log to the scratchpad. `PYTHONUNBUFFERED=1` is set for the
session, so the log is live. A block-buffered log read mid-run looks like a stalled job, and was
once diagnosed as one. For a Python script outside the CLI, pass `-u` as well.

## What things cost, measured on this machine

| job | time | footprint |
|---|---|---|
| reading the training half (once per table, then cached) | 3.6 min | ~1.0 GB on disk |
| `creditsurv select`, cold, every fit remade | ~21 min | ~2.8 GB across 4 workers |
| a report-grade fit with rule 14's shape, NumPy kernel path | ~16 min | ~1 GB |
| the same fit on the autograd path | 45-90 min | several GB |
| `creditsurv views` | ~80 min | 11-15 GB, alone on the machine |
| `creditsurv windows` (four fits, anchoring, grades) | 1.5-2 h | ~8 GB |
| `creditsurv moratorium` | ~4.8 h | 17 GB peak, swaps |
| streamed scoring of one window through `episode_hazards` | ~2.5-3 h, 453 batches | flat |

Machine speed drifts 13-18% within a session. Quote a ratio only against a baseline measured in
the same run.

## Monitoring

- Wait with an until-loop on a log condition (`until grep -q "<marker>" log; do sleep 30; done`),
  not with chained sleeps.
- **`ps -o etime` is `[[DD-]HH:]MM:SS`.** `12:14:50` is twelve **hours**. This was misread as
  minutes, and a job 31 hours old was taken for one 30 minutes old.
- Measure memory as phys_footprint ("peak memory footprint" from `/usr/bin/time -l`), not `maxrss`,
  which misses compressed pages and understates by about 2.5x.

## When a run is slower than 2x its budget

Diagnose before killing:
- `sample <pid> 3 -file <out>` shows where it is;
- `vm_stat` and `sysctl vm.swapusage` show whether it is paging. Swap-ins in the hundreds of
  millions mean paging, not computing, and no amount of waiting fixes it.

## Killing, and verifying the kill

Kill the wrapper and the Python child (`kill -TERM`, then `kill -9` whatever survives). Then
confirm that `pgrep -f "creditsurv <command>"` returns nothing. A queued runner whose child is
killed starts the next command: one restart left a log-logistic selection running beside a
prepayment one for three and a half hours, every fit at half speed. Check the newest cached fit's
`cause` and `distribution` too, since a fit landing with the wrong one gives away a hidden job.
