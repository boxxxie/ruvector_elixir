# Rustler NIF for RuvectorElixir

This directory contains the Rust native implemented functions (NIF) for [RuvectorElixir](https://github.com/boxxxie/ruvector_elixir).

## Overview

The NIF interfaces directly with the `ruvector-core` vector database engine written in Rust. Operations execute on Erlang VM dirty schedulers (`DirtyIo` and `DirtyCpu`) to prevent blocking normal Erlang schedulers during intensive disk I/O and approximate nearest neighbor (ANN) vector calculations.

## Building

The NIF is compiled automatically by [Rustler](https://github.com/rusterlium/rustler) during `mix compile`. A local Rust toolchain (`cargo` and `rustc` 1.91+) is required.

## Exports

- `open_db_with_options/3`, `close_db/1`, `info/1`
- `insert/2`, `insert_batch/2`
- `get_entry/2`, `delete/2`
- `db_len/1`, `is_empty/1`, `all_ids/1`
- `search/5`, `search_detailed/5`
- `calculate_distance/3`, `calculate_distance_dirty/3`
