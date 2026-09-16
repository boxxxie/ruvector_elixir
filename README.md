# RuvectorElixir

[![Hex.pm](https://img.shields.io/hexpm/v/ruvector_elixir.svg)](https://hex.pm/packages/ruvector_elixir)
[![HexDocs](https://img.shields.io/badge/hex-docs-purple.svg)](https://hexdocs.pm/ruvector_elixir)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official Elixir bindings for [**ruvector**](https://github.com/ruvnet/ruvector), an ultra-fast embedded vector database and GraphRAG metadata engine written in Rust.

`RuvectorElixir` allows you to store, index, and query high-dimensional vector embeddings with sub-millisecond approximate nearest neighbor (ANN) search directly from BEAM applications—without external services, daemon processes, or network overhead.

---

## Features

- **Embedded & Zero-Dependency**: Embedded vector storage with zero external database dependencies. Runs in-process inside your BEAM node.
- **High-Performance NIF**: Built with [Rustler](https://github.com/rusterlium/rustler). Heavy computation and disk I/O are safely executed on Erlang dirty schedulers (`DirtyCpu` and `DirtyIo`) to prevent scheduler starvation.
- **Multiple Distance Metrics**:
  - `:cosine` (Cosine Similarity)
  - `:euclidean` (Euclidean / L2 Distance)
  - `:dot_product` (Dot Product)
  - `:manhattan` (Manhattan / L1 Distance)
- **HNSW & Flat Indexing**: Graph-based Hierarchical Navigable Small World (HNSW) indexing for sub-millisecond retrieval on large datasets, plus exact flat indexing.
- **Rich JSON Metadata Filtering**: Filter vector search results using arbitrary JSON-serializable key-value metadata.
- **ACID Persistence**: Crash-safe on-disk embedded database files (`.rvf`).
- **Idiomatic Elixir**: First-class structs (`RuvectorElixir.VectorEntry`), comprehensive typespecs, and functional APIs.

---

## Installation

Add `ruvector_elixir` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:ruvector_elixir, "~> 0.1.0"}
  ]
end
```

### System Requirements

A working **Rust toolchain** (Rust 1.75+ or later) is required to compile the native extension:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

---

## Quickstart

### 1. Opening a Database

Create or open a database with a specified vector dimension (e.g. 128 dimensions):

```elixir
# Default Cosine metric with exact flat indexing
{:ok, db} = RuvectorElixir.open("priv/data/embeddings.rvf", 128)

# With HNSW index and Euclidean distance
{:ok, db} = RuvectorElixir.open("priv/data/embeddings.rvf", 128, %{
  metric: :euclidean,
  hnsw: %{
    m: 16,
    ef_construction: 100,
    ef_search: 50,
    max_elements: 100_000
  }
})
```

### 2. Inserting Vectors

Vectors can be inserted individually or in bulk using `RuvectorElixir.VectorEntry` structs, maps, or raw lists of floats:

```elixir
alias RuvectorElixir.VectorEntry

# Insert with explicit ID and metadata using VectorEntry
entry = VectorEntry.new([0.1, 0.2, 0.3, ...], id: "doc_1", metadata: %{
  "title" => "Elixir Guide",
  "category" => "programming",
  "views" => 1200
})
{:ok, "doc_1"} = RuvectorElixir.insert(db, entry)

# Insert with auto-generated UUID
{:ok, id} = RuvectorElixir.insert(db, [0.1, 0.2, 0.3, ...])

# Bulk insert
entries = [
  %{id: "doc_2", vector: [...], metadata: %{"category" => "news"}},
  %{id: "doc_3", vector: [...], metadata: %{"category" => "programming"}}
]
{:ok, ["doc_2", "doc_3"]} = RuvectorElixir.insert_batch(db, entries)
```

### 3. Searching for Nearest Neighbors

Perform top-`k` similarity search using a query vector:

```elixir
query = [0.1, 0.2, 0.3, ...]

# Fast ID-only search
ids = RuvectorElixir.search(db, query, 5)
# => ["doc_1", "doc_3", "doc_2"]

# Search with metadata filtering
filtered_ids = RuvectorElixir.search(db, query, %{"category" => "programming"}, 5)
# => ["doc_1", "doc_3"]

# Detailed search (returns ID, distance score, vector, and metadata)
{:ok, results} = RuvectorElixir.search_detailed(db, query, %{"category" => "programming"}, 5)
Enum.each(results, fn r ->
  IO.puts("ID: #{r.id} (Score: #{r.score}) - Title: #{r.metadata["title"]}")
end)
```

### 4. Fetching, Deleting & Inspecting

```elixir
# Retrieve vector and metadata by ID
{:ok, entry} = RuvectorElixir.get(db, "doc_1")
IO.inspect(entry.vector)
IO.inspect(entry.metadata)

# Check count and keys
{:ok, count} = RuvectorElixir.len(db)
{:ok, all_ids} = RuvectorElixir.keys(db)
{:ok, false} = RuvectorElixir.empty?(db)

# Delete an entry
{:ok, true} = RuvectorElixir.delete(db, "doc_1")

# Database info
info = RuvectorElixir.info(db)
# => %{"dimensions" => 128, "distance_metric" => "euclidean", "node_count" => 2}
```

### 5. Distance Calculations

Direct vector-to-vector distance calculations without opening a database:

```elixir
v1 = [1.0, 0.0, 0.0]
v2 = [0.0, 1.0, 0.0]

{:ok, dist} = RuvectorElixir.distance(v1, v2, :cosine)
# => {:ok, 1.0}

{:ok, dist} = RuvectorElixir.calculate_distance(v1, v2, :euclidean)
# => {:ok, 1.4142135}
```

---

## Configuration Options

### Database Options

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `:metric` | atom / string | `:cosine` | Distance metric (`:cosine`, `:euclidean`, `:dot_product`, `:manhattan`). |
| `:hnsw` | boolean / map | `nil` | `true` for default HNSW, or map with tuning parameters (see below). |

### HNSW Configuration

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `:m` | pos_integer | `16` | Maximum number of outgoing edges per node in the graph. |
| `:ef_construction` | pos_integer | `100` | Size of the candidate list evaluated during index construction. |
| `:ef_search` | pos_integer | `50` | Default candidate list size during search. |
| `:max_elements` | pos_integer | `100_000` | Initial pre-allocated capacity for vectors. |

---

## Architecture

`RuvectorElixir` communicates with `ruvector-core` via a native C-ABI bridge built with Rustler:

```
+-------------------------------------------------------------+
|                      Elixir Application                     |
|                   RuvectorElixir (BEAM)                     |
+------------------------------+------------------------------+
                               |
                   Rustler NIF Interface
                               |
               +---------------+---------------+
               |                               |
       [DirtyIo Threads]               [DirtyCpu Threads]
     - open_db / persistence         - k-NN search
     - insert / insert_batch         - distance calculations
     - get / delete / all_ids        - HNSW graph traversal
               |                               |
               +---------------+---------------+
                               |
                    ruvector-core (Rust)
                    - VectorDB & Sled Storage
                    - HNSW & Flat Indexing
                    - SIMD Distance Metrics
```

- **Thread Safety**: The Rust database engine features concurrent storage and internal index locking, managed directly inside a Rustler `ResourceArc`. Multiple Elixir processes (e.g. concurrent Tasks or GenServers) can concurrently read, insert, and search the database with zero locking overhead.
- **Zero BEAM Blocking**: All CPU-intensive searches run on Erlang `DirtyCpu` schedulers, while all disk I/O operations run on `DirtyIo` schedulers, keeping the normal BEAM schedulers free for Web / LiveView requests.
- **Idiomatic APIs & Bang Variants**: Supports both tuple results `{:ok, res} | {:error, reason}` and bang variants (`open!`, `insert!`, `search!`, `get!`, `delete!`, `len!`) that raise on failure. Options can be supplied as maps or keyword lists.

---

## Performance & Benchmarks

Because `ruvector` leverages SIMD instructions (AVX-512, AVX2, NEON) in Rust, vector operations achieve sub-millisecond query latencies:

- **Flat Search**: Exact search over 10,000 128-dimensional vectors in ~1.5ms.
- **HNSW Search**: Approximate search over 100,000+ vectors in < 0.2ms with >99% recall.
- **Batch Insertion**: Up to 100,000 vectors/sec with concurrent pipeline insertion.

---

## Repository & Development

This repository tracks the official upstream [ruvector](https://github.com/ruvnet/ruvector) repository as a Git submodule located at `reference/ruvector`.

### Cloning with Submodule

```bash
git clone --recurse-submodules https://github.com/ruvnet/ruvector.git
# Or if already cloned:
git submodule update --init --recursive
```

### Running Tests

Execute the complete test suite:

```bash
mix test
```

Generate ExDoc documentation:

```bash
mix docs
```

---

## Upstream Project

- **Core Repository**: [https://github.com/ruvnet/ruvector](https://github.com/ruvnet/ruvector)
- **Crates**: `ruvector-core`, `ruvector-graph`

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
