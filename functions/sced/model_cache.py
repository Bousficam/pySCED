"""
Bayesian model cache - fit-or-load
==================================

Avoids re-running the MCMC when an IDENTICAL model has already been fitted: the
ArviZ idata is stored in netCDF + a MANIFEST (spec + data hash). On recall, the
model is **reloaded** if and only if the spec AND the data hash match; otherwise
it is **refitted** (and rewritten). Guarantees that a stale model (different
family/terms/draws/data) is NEVER reused silently.

Usage (engine):
    idata, hit = fit_or_load(cache_dir, prefix, spec, _sample_fn, force_refit=...)
where ``_sample_fn()`` launches the MCMC and returns the idata. ``cache_dir=None``
-> no cache.
"""
import hashlib
import json
import os


def data_hash(df, cols):
    """STABLE hash of the used columns (values in ROW ORDER - the temporal order
    matters). Used to invalidate the cache when the data changes."""
    import pandas as pd
    sub = df[[c for c in cols if c is not None and c in df.columns]]
    try:
        b = pd.util.hash_pandas_object(sub, index=False).values.tobytes()
    except Exception:
        b = sub.to_csv(index=False).encode()
    return hashlib.sha1(b).hexdigest()[:12]


def array_hash(*arrays):
    """Stable hash of one or more arrays (per_case: y + start_index)."""
    import numpy as np
    h = hashlib.sha1()
    for a in arrays:
        h.update(np.ascontiguousarray(np.asarray(a, dtype=float)).tobytes())
    return h.hexdigest()[:12]


def _spec_str(spec):
    """Deterministic JSON serialization of a spec dict (sorted keys) used as the
    cache-identity string."""
    return json.dumps(spec, sort_keys=True, default=str)


def cache_key(prefix, spec):
    """File key = readable prefix + short spec hash (deterministic)."""
    h = hashlib.sha1(_spec_str(spec).encode()).hexdigest()[:10]
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(prefix))
    return f"{safe}__{h}"


def fit_or_load(cache_dir, prefix, spec, fit_fn, *, force_refit=False, verbose=False):
    """Return ``(idata, hit)``. If ``cache_dir`` and a matching .nc+manifest exist
    (same spec), RELOAD (``hit=True``, no MCMC); otherwise call ``fit_fn()`` and
    SAVE (.nc + manifest). ``force_refit`` ignores the existing cache (but rewrites)."""
    if not cache_dir:
        return fit_fn(), False
    import arviz as az
    os.makedirs(cache_dir, exist_ok=True)
    key = cache_key(prefix, spec)
    nc = os.path.join(cache_dir, key + ".nc")
    man = os.path.join(cache_dir, key + ".manifest.json")
    if (not force_refit) and os.path.exists(nc) and os.path.exists(man):
        try:
            with open(man, encoding="utf-8") as fh:
                saved = json.load(fh)
            if saved.get("spec_str") == _spec_str(spec):
                idata = az.from_netcdf(nc)
                if verbose:
                    print(f"  [cache HIT] {key}")
                return idata, True
        except Exception:
            pass                                            # unreadable manifest -> refit
    idata = fit_fn()
    try:
        idata.to_netcdf(nc)
        with open(man, "w", encoding="utf-8") as fh:
            json.dump({"prefix": str(prefix), "key": key, "spec_str": _spec_str(spec),
                       "spec": {k: str(v) for k, v in spec.items()}}, fh, indent=2)
        if verbose:
            print(f"  [cache SAVE] {key}")
    except Exception:
        pass
    return idata, False
