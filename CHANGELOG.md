# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.2] - 2026-11-24

### Changed

- Switched `RuvectorElixir.Native` to `RustlerPrecompiled` (`~> 0.9`, NIF `2.16`) with precompiled binaries for `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu`, `x86_64-apple-darwin`, and `aarch64-apple-darwin`, making `rustler` optional at compile time (`RUVECTOR_BUILD=true` to force local source builds).

## [0.2.1] - 2026-11-21

### Changed

- Upgraded `ruvector-core` from `2.3.0` to `2.5.2` (bringing `FlatGraph`-based HNSW, `Weak`-backed storage pool lifecycle, and upstream recall fixes) and refreshed compatible Cargo dependencies (`redb 2.6.4`, `hnsw_rs 0.3.5`, `serde_json 1.0.154`, `uuid 1.29.3`).
- Upgraded the Rust `rustler` NIF crate from `0.38.0` to `0.39.0` (retaining Rust `1.91` MSRV and Mix `rustler ~> 0.38.0`).
- Added `QuantizationConfig::Turbo4` encoding in `info/1` and updated the `db_info` `:quantization` typespec and docs.

## [0.2.0] - 2026-11-21

### Breaking Changes

- Upgraded Rustler dependency to `~> 0.38.0` (`runtime: false`), raised the minimum Elixir version to `~> 1.18` (CI-tested through Elixir `1.20.5` / Erlang/OTP `29.1.2`), and set the Rust MSRV to `1.91`.
- Moved low-level NIF stubs to the internal NIF bridge module (RuvectorElixir.Native) and removed legacy top-level aliases (`open_db/2,3`, `get_entry/2`, `db_len/1`, `is_empty/1`, `all_ids/1`, `calculate_distance/3`).
- Standardized public API return shapes:
  - `search/3,4` returns `{:ok, ids}` instead of detailed result maps (use `search_detailed/3,4` for scores, vectors, and metadata).
  - `empty?/1` returns a bare `boolean()` (raising `RuvectorElixir.Error` on closed or errored handles).
  - `info/1` returns `{:ok, map}` with atom keys (`:dimensions`, `:node_count`, `:distance_metric`, `:storage_path`, `:hnsw`, `:quantization`).
  - `search_detailed/3,4` now takes a trailing `opts` keyword list or map supporting `:filter` and `:ef_search`.
- Added `RuvectorElixir.Error`: bang (`!`) variants raise `RuvectorElixir.Error` on operational or NIF failures and `ArgumentError` on invalid caller inputs or options.
- Hardened boundary validation across Elixir and Rust: rejects unknown or duplicate options, invalid filters, structs/tuples/PIDs/non-UTF-8 binaries in metadata, `nil`/boolean/empty IDs, improper lists, and non-finite or `f32`-overflowing vectors rather than silently falling back.

### Added

- Isolated in-memory databases via `"memory://<name>"` URIs (`InMemoryDB`) that do not touch the filesystem.
- Explicit handle lifecycle functions: `close/1`, `close!/1`, `with_open/3,4`, and `with_open!/3,4`, with idempotent close and `{:error, :closed}` on subsequent calls.
- Per-query `:ef_search` tuning option for `search/4`, `search!/4`, `search_detailed/4`, and `search_detailed!/4`.
- `usage-rules.md` guidance for AI agents and Hex consumers, a `mix precommit` verification alias, and a GitHub Actions CI matrix across Elixir `1.18`/OTP `27`, `1.19`/OTP `28`, and `1.20.5`/OTP `29.1.2`.

### Changed

- Migrated `DbResource` to `#[rustler::resource_impl]`, removed unused `ruvector-graph`, and routed all database I/O (`DirtyIo`) and search / large-vector distance (`DirtyCpu`) calls onto Erlang dirty schedulers.
- Encoded BEAM maps directly via `Term::map_from_term_arrays` / `rustler::term_map!` and canonicalized `VectorEntry.new/2` metadata atom keys to UTF-8 strings so `get_struct!/2` round-trips cleanly.
- Updated Dialyzer typespecs across `RuvectorElixir`, `RuvectorElixir.VectorEntry`, and the internal `Native` NIF module.

### Fixed

- Bounded HNSW parameters (`m: 2..256`, `ef_construction`/`ef_search`: `1..1_000_000`, `max_elements`: `1..10_000_000`), clamped search `k` to live entry count, and enforced platform `usize` bounds to prevent OOM aborts or decode mismatches.
- Guarded metric-specific `f32` vector norms and L1 bounds on insert and search so large finite vectors cannot overflow distances to `Inf`/`NaN` and poison subsequent queries.
- Fixed persistent HNSW upsert stale-score ghost results by deleting pre-existing IDs before re-insert under a serialization lock; fixed `InMemoryDB` atomic locking, batch deduplication, and `delete`/`insert_batch` index-before-entries ordering.
- Improved filtered and HNSW search recall via iterative over-fetching, Elixir-consistent `1 == 1.0` numeric matching, exact `i64`/`u64` equality, iterative `json_values_equal` with uniform `127`-container JSON depth caps, and proper-list decoding without Rustler `ListIterator` panics.
- Released the `RwLock` write guard before dropping database state during `close/1` teardown so concurrent callers immediately receive `{:error, :closed}`.

## [0.1.0] - 2026-09-16

### Added

- Initial Elixir NIF bindings for `ruvector-core` with flat and HNSW indexing, persistence (`.rvf`), metadata filtering, direct distance calculations, and `RuvectorElixir.VectorEntry`.

[0.2.2]: https://github.com/boxxxie/ruvector_elixir/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/boxxxie/ruvector_elixir/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/boxxxie/ruvector_elixir/releases/tag/v0.2.0
[0.1.0]: https://github.com/boxxxie/ruvector_elixir/releases/tag/v0.1.0
