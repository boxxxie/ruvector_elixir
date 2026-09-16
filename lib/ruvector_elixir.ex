defmodule RuvectorElixir do
  @moduledoc """
  High-performance Elixir NIF bindings to `ruvector-core`, an embedded vector database written in Rust.

  Provides sub-millisecond approximate nearest neighbor (ANN) search via HNSW or exact flat indexing,
  supporting multiple distance metrics (cosine, euclidean, dot product, manhattan), metadata filtering,
  and ACID-compliant on-disk persistence as well as pure in-memory databases (`memory://`).

  ## Architecture

  `RuvectorElixir` uses [Rustler](https://github.com/rusterlium/rustler) to interface directly with
  the `ruvector-core` Rust engine via dirty schedulers (`DirtyIo` and `DirtyCpu`) to avoid blocking
  the Erlang VM's normal scheduler threads.

  ## Features

  - **Vector Operations**: Insertion (single & batch), retrieval by ID, deletion, key enumeration.
  - **Distance Metrics**: Cosine similarity, Euclidean (L2), Dot product, Manhattan (L1).
  - **Search**: Fast k-nearest neighbor search with optional metadata JSON filtering and `ef_search` parameter.
  - **HNSW Indexing**: Configurable `m`, `ef_construction`, and `ef_search` parameters for graph-based ANN indexing.
  - **In-Memory & Persistent Storage**: Sled/disk backed `.rvf` storage or pure in-memory `memory://<name>` databases.
  - **Quantization**: Support for scalar and binary quantization configurations.
  """

  use Rustler, otp_app: :ruvector_elixir, crate: "ruvector_nif"

  alias RuvectorElixir.VectorEntry

  @type db_ref :: reference()
  @type vector :: [number()]
  @type id_input :: String.t() | atom() | integer()
  @type metric :: :cosine | :euclidean | :dot_product | :dot | :manhattan | String.t()
  @type hnsw_options :: %{
          optional(:m) => pos_integer(),
          optional(:ef_construction) => pos_integer(),
          optional(:ef_search) => pos_integer(),
          optional(:max_elements) => pos_integer()
        }
  @type quantization_options ::
          :scalar
          | :binary
          | :none
          | String.t()
          | %{
              optional(:type) => :scalar | :binary | :product | String.t(),
              optional(:subspaces) => pos_integer(),
              optional(:k) => pos_integer()
            }
  @type db_options :: %{
          optional(:metric) => metric(),
          optional(:distance_metric) => metric(),
          optional(:hnsw) => boolean() | hnsw_options() | keyword(),
          optional(:quantization) => quantization_options() | keyword()
        }
  @type entry_map :: %{
          id: String.t(),
          vector: [float()],
          metadata: map() | nil
        }
  @type entry_input ::
          VectorEntry.t()
          | %{
              optional(:id) => id_input(),
              optional(:vector) => vector(),
              optional(:metadata) => map() | nil
            }
          | vector()

  @type search_result :: %{
          id: String.t(),
          score: float(),
          vector: [float()] | nil,
          metadata: map() | nil
        }

  # ============================================================================
  # Database Lifecycle
  # ============================================================================

  @doc """
  Opens or creates a ruvector database at `path` with the specified vector `dimensions` and optional configuration `options`.

  Options can be supplied as either a map or a keyword list.

  ## Options

  - `:metric` / `:distance_metric` - Distance metric to use (`:cosine`, `:euclidean`, `:dot_product`, `:manhattan`). Default: `:cosine`.
  - `:hnsw` - Either a boolean `true` for default HNSW index parameters, or a map/keyword list with:
    - `:m` - Number of bi-directional links created per node (default: 16).
    - `:ef_construction` - Size of dynamic candidate list during construction (default: 100).
    - `:ef_search` - Size of dynamic candidate list during search (default: 50).
    - `:max_elements` - Initial capacity of the graph index (default: 1_000_000).
  - `:quantization` - Optional quantization configuration (`:scalar`, `:binary`, `:none`, or product map).

  ## In-Memory Databases

  Use a path prefixed with `memory://` (e.g. `memory://my_session`) for a non-persisted in-memory database.

  ## Examples

      {:ok, db} = RuvectorElixir.open("my_db.rvf", 128)
      {:ok, db} = RuvectorElixir.open("my_db.rvf", 128, metric: :euclidean, hnsw: true)
      {:ok, mem_db} = RuvectorElixir.open("memory://transient", 64)

  """
  @spec open(Path.t(), pos_integer(), db_options() | keyword()) ::
          {:ok, db_ref()} | {:error, String.t()}
  def open(path, dimensions, options \\ %{})
      when is_binary(path) and is_integer(dimensions) and dimensions > 0 do
    open_db_with_options(path, dimensions, normalize_options(options))
  end

  @doc """
  Opens a database, raising `RuntimeError` on failure.
  """
  @spec open!(Path.t(), pos_integer(), db_options() | keyword()) :: db_ref()
  def open!(path, dimensions, options \\ %{}) do
    case open(path, dimensions, options) do
      {:ok, db} -> db
      {:error, reason} -> raise RuntimeError, "Failed to open database: #{reason}"
    end
  end

  @doc """
  Opens a ruvector database at `path` with `dimensions`.

  Equivalent to `open(path, dimensions, %{})`.
  """
  @spec open_db(Path.t(), pos_integer()) :: {:ok, db_ref()} | {:error, String.t()}
  def open_db(_path, _dimensions), do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Opens a ruvector database with configuration options.
  """
  @spec open_db(Path.t(), pos_integer(), db_options() | keyword()) ::
          {:ok, db_ref()} | {:error, String.t()}
  def open_db(path, dimensions, options), do: open(path, dimensions, options)

  @doc """
  Native NIF: Opens a ruvector database with optional settings map.
  """
  @spec open_db_with_options(Path.t(), pos_integer(), db_options() | nil) ::
          {:ok, db_ref()} | {:error, String.t()}
  def open_db_with_options(_path, _dimensions, _options), do: :erlang.nif_error(:nif_not_loaded)

  # ============================================================================
  # Insert & Storage Operations
  # ============================================================================

  @doc """
  Inserts a vector entry into the database.

  The entry can be:
  - A `RuvectorElixir.VectorEntry` struct.
  - A map with `:vector` (and optional `:id`, `:metadata`).
  - A raw list of numbers `[1.0, 2.0, ...]` (in which case a UUID is automatically assigned).

  Returns `{:ok, id}` on success, or `{:error, reason}` on failure.

  ## Examples

      {:ok, id} = RuvectorElixir.insert(db, [0.1, 0.2, 0.3])
      {:ok, "doc_1"} = RuvectorElixir.insert(db, %{id: "doc_1", vector: [0.1, 0.2, 0.3], metadata: %{"author" => "Alice"}})

  """
  @spec insert(db_ref(), entry_input()) :: {:ok, String.t()} | {:error, String.t()}
  def insert(_db_ref, _entry), do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Inserts a vector entry into the database, raising `ArgumentError` on failure.
  """
  @spec insert!(db_ref(), entry_input()) :: String.t()
  def insert!(db_ref, entry) do
    case insert(db_ref, entry) do
      {:ok, id} -> id
      {:error, reason} -> raise ArgumentError, reason
    end
  end

  @doc """
  Inserts a batch of vector entries into the database.

  Returns `{:ok, [id]}` on success, or `{:error, reason}` on failure.

  ## Examples

      entries = [
        %{id: "v1", vector: [0.1, 0.2, 0.3]},
        %{id: "v2", vector: [0.4, 0.5, 0.6]}
      ]
      {:ok, ["v1", "v2"]} = RuvectorElixir.insert_batch(db, entries)

  """
  @spec insert_batch(db_ref(), [entry_input()]) :: {:ok, [String.t()]} | {:error, String.t()}
  def insert_batch(_db_ref, _entries), do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Inserts a batch of vector entries into the database, raising `ArgumentError` on failure.
  """
  @spec insert_batch!(db_ref(), [entry_input()]) :: [String.t()]
  def insert_batch!(db_ref, entries) do
    case insert_batch(db_ref, entries) do
      {:ok, ids} -> ids
      {:error, reason} -> raise ArgumentError, reason
    end
  end

  @doc """
  Retrieves a vector entry by its ID (string, atom, or integer).

  Returns `{:ok, %{id: id, vector: vector, metadata: metadata}}` if found,
  `{:error, :not_found}` if no entry with that ID exists, or `{:error, reason}` on failure.
  """
  @spec get_entry(db_ref(), id_input()) ::
          {:ok, entry_map()} | {:error, :not_found} | {:error, String.t()}
  def get_entry(_db_ref, _id), do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Retrieves a vector entry, raising `RuntimeError` if not found or if an error occurs.
  """
  @spec get_entry!(db_ref(), id_input()) :: entry_map()
  def get_entry!(db_ref, id) do
    case get_entry(db_ref, id) do
      {:ok, entry} -> entry
      {:error, :not_found} -> raise RuntimeError, "Vector entry with id #{inspect(id)} not found"
      {:error, reason} -> raise RuntimeError, reason
    end
  end

  @doc """
  Convenience alias for `get_entry/2`.
  """
  @spec get(db_ref(), id_input()) ::
          {:ok, entry_map()} | {:error, :not_found} | {:error, String.t()}
  def get(db_ref, id), do: get_entry(db_ref, id)

  @doc """
  Convenience alias for `get_entry!/2`.
  """
  @spec get!(db_ref(), id_input()) :: entry_map()
  def get!(db_ref, id), do: get_entry!(db_ref, id)

  @doc """
  Retrieves a vector entry and returns it as a `RuvectorElixir.VectorEntry` struct.
  """
  @spec get_struct(db_ref(), id_input()) ::
          {:ok, VectorEntry.t()} | {:error, :not_found} | {:error, String.t()}
  def get_struct(db_ref, id) do
    case get_entry(db_ref, id) do
      {:ok, entry_map} -> {:ok, VectorEntry.from_map(entry_map)}
      error -> error
    end
  end

  @doc """
  Retrieves a vector entry as a `VectorEntry` struct, raising `RuntimeError` if not found or on error.
  """
  @spec get_struct!(db_ref(), id_input()) :: VectorEntry.t()
  def get_struct!(db_ref, id) do
    case get_struct(db_ref, id) do
      {:ok, struct} -> struct
      {:error, :not_found} -> raise RuntimeError, "Vector entry with id #{inspect(id)} not found"
      {:error, reason} -> raise RuntimeError, reason
    end
  end

  @doc """
  Deletes an entry by its ID (string, atom, or integer).

  Returns `{:ok, true}` if deleted, `{:ok, false}` if not found, or `{:error, reason}` on failure.
  """
  @spec delete(db_ref(), id_input()) :: {:ok, boolean()} | {:error, String.t()}
  def delete(_db_ref, _id), do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Deletes an entry by its ID, raising `RuntimeError` on failure.
  """
  @spec delete!(db_ref(), id_input()) :: boolean()
  def delete!(db_ref, id) do
    case delete(db_ref, id) do
      {:ok, deleted} -> deleted
      {:error, reason} -> raise RuntimeError, reason
    end
  end

  @doc """
  Returns the total number of vectors in the database.
  """
  @spec db_len(db_ref()) :: {:ok, non_neg_integer()} | {:error, String.t()}
  def db_len(_db_ref), do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Convenience alias for `db_len/1`.
  """
  @spec len(db_ref()) :: {:ok, non_neg_integer()} | {:error, String.t()}
  def len(db_ref), do: db_len(db_ref)

  @doc """
  Returns the total number of vectors in the database, raising on error.
  """
  @spec len!(db_ref()) :: non_neg_integer()
  def len!(db_ref) do
    case len(db_ref) do
      {:ok, count} -> count
      {:error, reason} -> raise RuntimeError, reason
    end
  end

  @doc """
  Convenience alias for `db_len/1`.
  """
  @spec count(db_ref()) :: {:ok, non_neg_integer()} | {:error, String.t()}
  def count(db_ref), do: db_len(db_ref)

  @doc """
  Convenience alias for `len!/1`.
  """
  @spec count!(db_ref()) :: non_neg_integer()
  def count!(db_ref), do: len!(db_ref)

  @doc """
  Checks if the database is empty.
  """
  @spec is_empty(db_ref()) :: {:ok, boolean()} | {:error, String.t()}
  def is_empty(_db_ref), do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Convenience alias for `is_empty/1`.
  """
  @spec empty?(db_ref()) :: {:ok, boolean()} | {:error, String.t()}
  def empty?(db_ref), do: is_empty(db_ref)

  @doc """
  Returns a list of all vector IDs currently stored in the database.
  """
  @spec all_ids(db_ref()) :: {:ok, [String.t()]} | {:error, String.t()}
  def all_ids(_db_ref), do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Convenience alias for `all_ids/1`.
  """
  @spec keys(db_ref()) :: {:ok, [String.t()]} | {:error, String.t()}
  def keys(db_ref), do: all_ids(db_ref)

  @doc """
  Returns all vector IDs in the database, raising on error.
  """
  @spec keys!(db_ref()) :: [String.t()]
  def keys!(db_ref) do
    case keys(db_ref) do
      {:ok, ids} -> ids
      {:error, reason} -> raise RuntimeError, reason
    end
  end

  # ============================================================================
  # Search Operations
  # ============================================================================

  @doc """
  Searches the database for the top `k` closest vectors to `query_vector`.

  Returns a list of matching vector ID strings, or `{:error, reason}` on failure.
  """
  @spec search(db_ref(), vector(), pos_integer()) :: [String.t()] | {:error, String.t()}
  def search(db_ref, query_vector, k), do: search(db_ref, query_vector, nil, k)

  @doc """
  Searches the database for the top `k` closest vectors matching the optional `filter`.

  Filter can be a map or a keyword list.

  Returns a list of matching vector ID strings, or `{:error, reason}` on failure.
  """
  @spec search(db_ref(), vector(), map() | keyword() | nil, pos_integer()) ::
          [String.t()] | {:error, String.t()}
  def search(_db_ref, _query_vector, _filter, _k), do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Searches the database for top `k` closest vectors, raising `ArgumentError` on failure.
  """
  @spec search!(db_ref(), vector(), pos_integer()) :: [String.t()]
  def search!(db_ref, query_vector, k) do
    search!(db_ref, query_vector, nil, k)
  end

  @doc """
  Searches the database for top `k` closest vectors with filter, raising `ArgumentError` on failure.
  """
  @spec search!(db_ref(), vector(), map() | keyword() | nil, pos_integer()) :: [String.t()]
  def search!(db_ref, query_vector, filter, k) do
    case search(db_ref, query_vector, filter, k) do
      {:error, reason} -> raise ArgumentError, reason
      ids when is_list(ids) -> ids
    end
  end

  @doc """
  Searches the database for top `k` closest vectors, returning detailed search result maps.

  Each result map contains:
  - `:id` - Vector ID string
  - `:score` - Similarity / distance score (lower is closer for distance metrics)
  - `:vector` - Vector embedding float list (or nil)
  - `:metadata` - Metadata map (or nil)
  """
  @spec search_detailed(db_ref(), vector(), pos_integer()) ::
          {:ok, [search_result()]} | {:error, String.t()}
  def search_detailed(db_ref, query_vector, k) do
    search_detailed(db_ref, query_vector, nil, k, nil)
  end

  @doc """
  Searches the database with a metadata filter, returning detailed search results.
  """
  @spec search_detailed(db_ref(), vector(), map() | keyword() | nil, pos_integer()) ::
          {:ok, [search_result()]} | {:error, String.t()}
  def search_detailed(db_ref, query_vector, filter, k) do
    search_detailed(db_ref, query_vector, filter, k, nil)
  end

  @doc """
  Native NIF: Searches the database with a filter and optional HNSW `ef_search` override.
  """
  @spec search_detailed(
          db_ref(),
          vector(),
          map() | keyword() | nil,
          pos_integer(),
          pos_integer() | nil
        ) ::
          {:ok, [search_result()]} | {:error, String.t()}
  def search_detailed(_db_ref, _query_vector, _filter, _k, _ef_search) do
    :erlang.nif_error(:nif_not_loaded)
  end

  @doc """
  Searches the database returning detailed search results, raising `ArgumentError` on failure.
  """
  @spec search_detailed!(
          db_ref(),
          vector(),
          map() | keyword() | nil,
          pos_integer(),
          pos_integer() | nil
        ) :: [search_result()]
  def search_detailed!(db_ref, query_vector, filter \\ nil, k, ef_search \\ nil) do
    case search_detailed(db_ref, query_vector, filter, k, ef_search) do
      {:ok, results} -> results
      {:error, reason} -> raise ArgumentError, reason
    end
  end

  # ============================================================================
  # Distance Utilities & Info
  # ============================================================================

  @doc """
  Calculates the distance between two vectors `a` and `b` using the specified `metric`.

  Supported metrics: `:cosine`, `:euclidean`, `:dot_product`, `:manhattan`.

  Returns `{:ok, distance}` on success, or `{:error, reason}` on failure (e.g. dimension mismatch).
  """
  @spec calculate_distance(vector(), vector(), metric()) ::
          {:ok, float()} | {:error, String.t()}
  def calculate_distance(_a, _b, _metric), do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Convenience helper for `calculate_distance/3`, defaulting to `:cosine`.
  """
  @spec distance(vector(), vector(), metric()) :: {:ok, float()} | {:error, String.t()}
  def distance(a, b, metric \\ :cosine), do: calculate_distance(a, b, metric)

  @doc """
  Calculates distance between two vectors, raising `ArgumentError` on failure.
  """
  @spec distance!(vector(), vector(), metric()) :: float()
  def distance!(a, b, metric \\ :cosine) do
    case distance(a, b, metric) do
      {:ok, dist} -> dist
      {:error, reason} -> raise ArgumentError, reason
    end
  end

  @doc """
  Gets database metadata info including `:dimensions`, `:node_count`, `:distance_metric`, `:storage_path`, and `:hnsw`.
  """
  @spec info(db_ref()) :: map()
  def info(_db_ref), do: :erlang.nif_error(:nif_not_loaded)

  # ============================================================================
  # Helpers
  # ============================================================================

  defp normalize_options(opts) when is_list(opts) do
    opts
    |> Map.new()
    |> normalize_options()
  end

  defp normalize_options(%{} = map) do
    Enum.reduce(map, %{}, fn {k, v}, acc ->
      key = if is_atom(k), do: Atom.to_string(k), else: to_string(k)

      val =
        case {key, v} do
          {"hnsw", hnsw_opts} when is_list(hnsw_opts) ->
            Map.new(hnsw_opts, fn {hk, hv} ->
              {if(is_atom(hk), do: Atom.to_string(hk), else: to_string(hk)), hv}
            end)

          {"hnsw", %{} = hnsw_map} ->
            Map.new(hnsw_map, fn {hk, hv} ->
              {if(is_atom(hk), do: Atom.to_string(hk), else: to_string(hk)), hv}
            end)

          {"quantization", q_opts} when is_list(q_opts) ->
            Map.new(q_opts, fn {qk, qv} ->
              {if(is_atom(qk), do: Atom.to_string(qk), else: to_string(qk)), qv}
            end)

          _ ->
            v
        end

      Map.put(acc, key, val)
    end)
  end

  defp normalize_options(_), do: %{}
end
