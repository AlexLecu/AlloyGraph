# Superseded design artefacts

## `random_search_baseline_opt.json`

An earlier run of `random_search_baseline.py` at N = 5 carrying four arms:
the Designer reference, plain random search, plausibility-aware random search,
and `random+Guard+Tuner`. It is a strict subset of the current
`results/random_search_baseline.json`, which adds the fourth grid cell —
`random+Guard+Tuner (plaus-aware)` — that completes the 2x2 of
{plain, Guard+Tuner} x {select on hits, select on plausibility first}.

That missing cell mattered. It is the strongest baseline available at the
Designer's own budget, and it produces 1/20 credible designs rather than the
0/20 every weaker arm produces. Any statement that random search yields *no*
credible compositions at equal budget was drawing on this file's incomplete
grid.

Kept for provenance, regenerable from the same script and seed. Not
authoritative.
