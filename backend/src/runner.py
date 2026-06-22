"""Phase 5 — In-process job queue + single worker thread (BUILD_SPEC.md 8).

One GPU -> one job at a time. A dependency-free ``queue.Queue`` holds pending run configs and a
single daemon worker thread drains it, calling :func:`train.train_one` for each. No Celery, no
Redis, no database — run state lives entirely in ``runs/<run_id>/card.json`` (see ``store.py``).

Lifecycle of a run:
    enqueue(cfg)            -> create runs/<id>/, write card.json {status: queued}, push, return id
    worker pops the job     -> train_one(cfg, run_id) sets running -> done
    worker on exception     -> set status failed + error; always continue to the next job
"""

from __future__ import annotations

import itertools
import queue
import threading
import traceback

from . import config, store, train

# Hyperparameters that may be swept, mapped to the coercion applied to each value.
SWEEPABLE = {"hidden_size": int, "num_layers": int, "lookback": int, "lr": float}

# (run_id, full_cfg) tuples awaiting the single GPU worker.
_q: "queue.Queue[tuple[str, dict]]" = queue.Queue()
_worker_started = False
_start_lock = threading.Lock()


def _worker() -> None:
    """Drain the queue forever; one job at a time so a single GPU is never oversubscribed."""
    while True:
        run_id, cfg = _q.get()
        try:
            train.train_one(cfg, run_id=run_id)          # sets running -> done, writes the report
        except Exception as exc:                          # noqa: BLE001 - record and keep going
            traceback.print_exc()
            try:
                store.set_status(run_id, "failed", error=str(exc))
            except Exception:                             # pragma: no cover - best effort
                pass
        finally:
            _q.task_done()


def start_worker() -> None:
    """Start the single worker thread once (idempotent; call at FastAPI startup)."""
    global _worker_started
    with _start_lock:
        if not _worker_started:
            threading.Thread(target=_worker, name="train-worker", daemon=True).start()
            _worker_started = True


def _full_config(overrides: dict | None) -> dict:
    cfg = config.default_config()
    cfg.update(overrides or {})
    return cfg


def _unique_run_id(cfg: dict) -> str:
    """``store.make_run_id`` keyed by time+hidden+layers, de-duplicated if a folder exists."""
    base = store.make_run_id(cfg)
    run_id = base
    i = 2
    while store.run_dir(run_id).exists():
        run_id = f"{base}_{i}"
        i += 1
    return run_id


def enqueue(overrides: dict | None = None) -> str:
    """Create a queued run and push it to the worker. Returns the run_id immediately."""
    config.ensure_dirs()
    cfg = _full_config(overrides)
    run_id = _unique_run_id(cfg)
    store.new_card(run_id, cfg)
    _q.put((run_id, cfg))
    start_worker()
    return run_id


def enqueue_sweep(
    grids: dict[str, list], shared: dict | None = None, *, mode: str = "one_factor"
) -> list[str]:
    """Queue a sweep over one or more hyperparameters, all as tuning runs.

    ``grids`` maps a :data:`SWEEPABLE` key to the list of values to try, e.g.
    ``{"hidden_size": [64, 128, 256], "lr": [1e-3, 5e-4]}``. ``shared`` carries the held-constant
    config (stride/batch/etc.) so every run is the fair comparison the spec calls for.

    Modes:
      - ``"one_factor"``: vary one param at a time over its list, others at their ``shared`` value.
      - ``"grid"``: the cartesian product of every list (the full matrix).
    """
    shared = dict(shared or {})
    shared["is_final"] = False                            # a sweep is always tuning
    grids = {k: [SWEEPABLE[k](x) for x in v] for k, v in grids.items() if k in SWEEPABLE and v}
    run_ids: list[str] = []

    if mode == "grid":
        keys = list(grids)
        for combo in itertools.product(*(grids[k] for k in keys)):
            overrides = dict(shared)
            overrides.update(dict(zip(keys, combo)))
            run_ids.append(enqueue(overrides))
    else:                                                 # one_factor
        for key, values in grids.items():
            for v in values:
                overrides = dict(shared)
                overrides[key] = v
                run_ids.append(enqueue(overrides))
    return run_ids


def queue_depth() -> int:
    """Approximate number of jobs waiting (excludes the one currently training)."""
    return _q.qsize()
