# RuvectorElixir Usage Rules

## Core API Patterns

- **Open & Lifecycle**: Always open with `RuvectorElixir.open/2,3` (or `open!/2,3`) using positive integer `dimensions` and a non-empty UTF-8 path (or `"memory://<name>"` for an isolated per-handle in-memory store). Prefer `RuvectorElixir.with_open/3,4` (or `with_open!/3,4`) for scoped handles, or call `RuvectorElixir.close/1` when done with short-lived handles so underlying `VectorDB` resources are released immediately.
- **Uniform Return Tuples**:
  - `open/3`, `insert/2`, `insert_batch/2`, `get/2`, `get_struct/2`, `delete/2`, `len/1`, `count/1`, `keys/1`, `search/3,4`, `search_detailed/3,4`, `distance/2,3`, and `info/1` return `{:ok, value} | {:error, reason}`.
  - `empty?/1` returns a bare `boolean()` (raising `RuvectorElixir.Error` if the handle is closed or errored) so `if RuvectorElixir.empty?(db)` works directly.
  - `close/1` returns `:ok | {:error, reason}` and is idempotent; operations on a closed handle return `{:error, :closed}`.
  - Bang (`!`) variants raise `RuvectorElixir.Error` on operational/NIF failures and `ArgumentError` on invalid caller inputs/options.
- **Search Signatures**: Pass optional `:filter` and `:ef_search` in the trailing `opts` keyword list/map:
  - `RuvectorElixir.search(db, query, k, filter: %{"tenant" => "acme"})`
  - `RuvectorElixir.search_detailed(db, query, k, filter: [tenant: "acme"], ef_search: 100)`
- **Info Map Keys**: `info/1` returns atom keys (`:dimensions`, `:node_count`, `:distance_metric`, `:storage_path`, `:hnsw`, `:quantization`).

## Guardrails & Constraints

- **No Legacy Aliases**: Do not call `open_db/*`, `get_entry/*`, `db_len/1`, `is_empty/1`, `all_ids/1`, or `calculate_distance/3`, and do not call `RuvectorElixir.Native` directly.
- **In-Memory Isolation**: Every `RuvectorElixir.open("memory://...", dims)` call creates an independent in-memory database even if the same URI string is passed twice.
- **32-bit Float Precision**: Vectors are stored as Rust `f32`; non-dyadic 64-bit Elixir floats are rounded to `f32` precision on insert.
- **HNSW & Quantization Constraints**: `:hnsw` `:m` must be in `2..256`, `:ef_construction` and `:ef_search` in `1..1_000_000`, and `:max_elements` in `1..10_000_000`; `:quantization` `:subspaces` must evenly divide `dimensions`. Note that `ruvector-core` 2.3.0 persists/reports `:quantization` in config/info without compressing stored vectors yet.
- **Strict Metadata & Filters**: Metadata and filters accept JSON-compatible primitives (`nil`, booleans, finite floats, integers, UTF-8 strings, lists, plain maps) up to 127 nested JSON containers. `VectorEntry.new/2` canonicalizes atom metadata keys to UTF-8 strings so `get_struct!/2` round-trips cleanly. Structs, tuples, PIDs, non-UTF-8 binaries, and non-finite floats are rejected.
