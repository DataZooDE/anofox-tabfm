# bench — reproducible model × device × shape timings

Every performance number in `docs/` was produced by hand with `.timer on` in a
shell and pasted into a table. That makes them unauditable: you cannot tell
whether two rows came from the same build, the same context size, or even the
same device — and **a device that silently fell back to the CPU looks exactly
like one that did not**, which is how this repo's first GPU equivalence result
turned out to be CPU vs CPU reporting a perfect score.

So this harness enforces one rule, the same one
`tools/gpu_test/scenarios/README.md` states: **every measurement carries the
device that actually served it**, read from `tabfm_models()` after the query.
A cell whose `served_by` is not what was asked for is flagged, never quietly
averaged in, and the process exits non-zero so a mismatch cannot pass in CI.

## Use

```bash
# CPU baseline
python3 tools/bench/bench.py --duckdb ./build/release/duckdb \
    --models mitra,tabdpt --devices cpu --rows 100,1000,2500 --features 10

# a GPU, via its plugin
python3 tools/bench/bench.py --duckdb ./build/release/duckdb \
    --models mitra --devices rocm --ep-path ~/.cache/anofox-tabfm/runtime \
    --rows 100,2500 --precision bf16
```

One JSON object per line, so runs append and are queryable by the thing this
repo is:

```bash
python3 tools/bench/bench.py ... >> bench.jsonl
duckdb -c "SELECT model, served_by, rows, per_call_ms FROM 'bench.jsonl'
           WHERE ok ORDER BY model, rows"
```

## How the number is derived, and what it is not

The CLI reports no per-statement timing in csv mode, and timing one session
would fold in process startup, extension load, table build, session
construction, weight injection and any shape compile. On a 6.5 GB model that
dwarfs the inference being measured.

So each cell runs the **identical session twice** — once with N timed repeats,
once with none. Everything except the N inferences is common to both and
cancels; the difference divided by N is the warm per-inference cost. That is
what `per_call_ms` means, and `metric` says so in every record.

It is therefore:

- **warm**, excluding the first call — which on MIGraphX pays a shape-bucket
  compile of minutes. Reporting that as latency would be wrong by three orders
  of magnitude.
- **end-to-end through SQL**, including preprocessing and decode, not a bare
  forward pass. That is the number a user experiences.
- **null when unmeasurable** — if the difference is not positive the inference
  is lost in startup noise at that size, and the field is `null` rather than a
  suspiciously small number.

## Before you quote a number

Run against a **release** build. A debug build is ASan-instrumented and is
roughly an order of magnitude slower; the figures are internally consistent but
meaningless as absolute performance.

And read `served_by` before `per_call_ms`. Always.
