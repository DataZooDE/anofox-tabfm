# Pure-SQL model surface — plan

**Goal (jr, 2026-07-14):** operate the extension entirely in SQL — no external
JSON manifest anywhere. A model is either **built in** (used by name) or
**registered in pure SQL** (`CALL tabfm_register_model(...)`). Weight setup is
also pure SQL: the extension reads PyTorch `.ckpt` natively, so
`tabfm_download('tabpfn-v2')` needs no Python. The `anofox_tabfm_model_manifest`
setting is **removed**.

Decisions: built-ins + SQL registration · `CALL` function · native `.ckpt`
reading · remove the manifest setting.

## End state (what operating it looks like)

```sql
LOAD anofox_tabfm;
-- built-in models, zero config:
SELECT * FROM tabfm_classify('train','species', test:='score', model:='mitra');
-- your own model, pure SQL (the only external thing is the .onnx graph blob):
CALL tabfm_register_model(
  id := 'my_model', family := 'icl-transformer',
  classification_graph := 'https://.../graph.onnx',
  weights := 'https://huggingface.co/acme/my/resolve/main/model.safetensors',
  tensor_map := MAP{'m.w':'w', ...},           -- or a path/url
  license := 'apache-2.0', preprocessing_profile := 'tabfm_v1_minimal',
  max_rows := 10000, max_features := 100, max_classes := 10);
SELECT * FROM tabfm_classify('train','y', test:='score', model:='my_model');
```

## Phase 1 — Built-in catalog (bake in the shipped models)

Promote Mitra / TabPFN v2 / TabICL v2 to the same status as `tabfm-v1`.

- **Bundle** their weight-free graphs + tensor-maps (both tasks) into the binary
  via `cmake/embed_resources.cmake` (extend the resource list; new bundled ids
  `graph_mitra_classification`, `tensor_map_tabpfn_regression`, …).
- **Bake specs** into `BuiltinModelSpecs` (`src/tabfm_registry.cpp`): translate
  `examples/{mitra,tabpfn,tabicl}.json` into C++ — id, license object,
  capabilities, size_regime, tensor_contract, preprocessing_profile, per-task
  artifacts (bundled graph id + bundled/per-task tensor_map + HF repo/files/urls).
- Result: `model := 'mitra'|'tabpfn-v2'|'tabicl-v2'` works with no config;
  `tabfm_list_models()` shows four built-ins by default.
- Tests: offline sqllogictest asserting the four built-ins list + capabilities +
  selection precedence (no weights needed until a predict).

## Phase 2 — `CALL tabfm_register_model(...)` (custom models + fixtures, in SQL)

- New function registering a model into the **db-instance registry** (TabFMState),
  session-scoped (re-register per script; no persistence file). Named args cover
  the full ModelSpec; `tensor_map` accepts a `MAP{onnx→st}` or a path/url; graphs
  accept path or url; optional `tabfm_unregister_model(id)`.
- `ModelRegistry::Build()` now merges **built-ins + SQL-registered specs** — the
  manifest source argument is gone.
- Tests: migrate every offline fixture test (`tabfm_mitra`, `tabfm_tabpfn`,
  `tabfm_tabicl`, `tabfm_contract`, `tabfm_registry_meta`, `tabfm_multimodel*`,
  `tabfm_weights`) from `SET anofox_tabfm_model_manifest = '<json>'` to
  `CALL tabfm_register_model(... classification_graph := 'test/fixtures/.../g.onnx' ...)`.
  (The register function is what makes the offline fixtures work without JSON.)

## Phase 3 — Native torch `.ckpt` reading (drop the Python converter)

The one remaining non-SQL step. TabPFN/TabICL ship pickle `.ckpt`; teach the
engine to read it.

- Minimal torch-save reader in C++: unzip the `.ckpt` (miniz), run a **subset
  pickle VM** (GLOBAL/REDUCE/BINPERSID/dict/OrderedDict/tuple/int/str +
  `_rebuild_tensor_v2`) to recover `name → (storage-key, dtype, shape, offset,
  stride)`, read the `data/<key>` blobs → the same in-memory tensor set the
  safetensors path yields.
- Wire into the weights loader: sniff the file (PK-zip + torch layout → ckpt;
  else safetensors) and inject.
- `tabfm_download('tabpfn-v2'|'tabicl-v2')` now fetches the HF `.ckpt` and injects
  directly. Retire `tools/export_*/convert_weights.py`.
- Risk: pickle correctness across torch versions, dtype/endianness, shared
  storages. Golden test: compare a handful of tensors to the safetensors the
  converter produced.
- **Biggest lift; highest risk.** Until it lands, the TabPFN/TabICL built-ins use
  the interim converter (documented).

## Phase 4 — Remove the manifest setting + migrate everything

- Delete `anofox_tabfm_model_manifest` (`tabfm_settings.cpp`) + its plumbing
  (`ModelRegistry::Build(manifest_source)`, `ResolveManifests`, `FindManifest`,
  the manifest read in `tabfm_list_models`).
- Update examples (drop the `SET`; built-ins or `CALL`), docs (README,
  REAL_MODELS.md), and retire `examples/*.json` (keep as a reference appendix only
  if useful).
- Full suite green; codex review; PR.

## Sequencing & risk

1. **Phase 1 + 2** deliver the pure-SQL surface (the core want) — medium effort,
   low risk. Check in here.
2. **Phase 3** (native ckpt) — high effort/risk; makes TabPFN/TabICL weight-setup
   pure-SQL. Interim converter documented until done.
3. **Phase 4** — mechanical but touches many tests; the hard break (removing the
   setting) lands last, once the replacements are proven.

Red/green TDD throughout; codex review after Phase 2 and Phase 3.
