defmodule RuvectorElixir do
  @moduledoc """
  High-performance Elixir NIF bindings to `ruvector-core`, an embedded vector database written in Rust.

  Provides sub-millisecond approximate nearest neighbor (ANN) search via HNSW or exact flat indexing,
  supporting multiple distance metrics (cosine, euclidean, dot product, manhattan), metadata filtering,
  on-disk persistence via `redb`, and isolated in-memory databases (`memory://`).

  ## Architecture

  `RuvectorElixir` uses [Rustler](https://github.com/rusterlium/rustler) to interface directly with
  `ruvector-core` on Erlang dirty schedulers (`DirtyIo` and `DirtyCpu`) so BEAM normal schedulers
  are never blocked.

  ## Features

  - **Vector Operations**: Insertion (single & batch), retrieval by ID, deletion, key enumeration.
  - **Distance Metrics**: Cosine similarity, Euclidean (L2), Dot product, Manhattan (L1).
  - **Search**: Fast k-nearest neighbor search with optional metadata filtering and `ef_search` tuning.
  - **HNSW Indexing**: Configurable `m`, `ef_construction`, and `ef_search` parameters for graph-based ANN indexing.
  - **In-Memory & Persistent Storage**: Disk-backed `.rvf` storage or pure in-memory `memory://<name>` databases.
  - **Explicit Lifecycle**: Deterministic `close/1` to release database handles immediately.
  """

  alias RuvectorElixir.{Error, Native, VectorEntry}

  @opaque db_ref :: reference()
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
              optional(:type) => :scalar | :binary | :none | :product | String.t(),
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
              required(:vector) => vector(),
              optional(:id) => id_input() | nil,
              optional(:metadata) => map() | nil
            }
          | vector()
  @type search_filter :: map() | keyword() | nil
  @type search_option :: {:filter, search_filter()} | {:ef_search, pos_integer() | nil}
  @type search_detailed_option :: search_option()
  @type search_result :: %{
          id: String.t(),
          score: float(),
          vector: [float()] | nil,
          metadata: map() | nil
        }
  @type db_info :: %{
          node_count: non_neg_integer(),
          dimensions: pos_integer(),
          distance_metric: :cosine | :euclidean | :dot_product | :manhattan,
          storage_path: String.t(),
          hnsw: hnsw_options() | nil,
          quantization:
            :none
            | :scalar
            | :binary
            | %{type: :product, subspaces: pos_integer(), k: pos_integer()}
            | nil
        }

  @valid_metrics [:cosine, :euclidean, :dot_product, :dot, :manhattan]
  @valid_metric_strings ["cosine", "euclidean", "dot_product", "dotproduct", "dot", "manhattan"]

  # ============================================================================
  # Database Lifecycle
  # ============================================================================

  @doc """
  Opens or creates a ruvector database at `path` with the specified vector `dimensions` and optional configuration `options`.

  Options can be supplied as either a plain map or a keyword list. Invalid option keys or values
  raise `ArgumentError`.

  ## Options

  - `:metric` / `:distance_metric` - Distance metric to use (`:cosine`, `:euclidean`, `:dot_product`, `:manhattan`). Default: `:cosine`.
  - `:hnsw` - Either a boolean (`true` for default HNSW index parameters, `false` for flat index), or a map/keyword list with:
    - `:m` - Number of bi-directional links created per node (`2..256`, default: `16`).
    - `:ef_construction` - Size of dynamic candidate list during construction (default: `100`).
    - `:ef_search` - Size of dynamic candidate list during search (default: `100`).
    - `:max_elements` - Initial capacity of the graph index (default: `1_000_000`).
  - `:quantization` - Optional quantization configuration (`:scalar`, `:binary`, `:none`, or `%{type: :product, subspaces: pos_integer(), k: pos_integer()}` where `subspaces` must evenly divide `dimensions`).
    Note: upstream `ruvector-core` 2.3.0 persists and reports `:quantization` in config/info, but stores vectors in `f32` precision without applying compression yet (`ruvnet/RuVector#563`).

  Vectors are stored internally as 32-bit IEEE 754 floats (`f32`); non-dyadic 64-bit Elixir floats
  are rounded to `f32` precision on insert and widened back to `f64` on read.

  ## In-Memory Databases

  Use a path prefixed with `memory://` (e.g. `"memory://transient"`) for a pure in-memory database
  that does not touch the filesystem. Every `open("memory://...", ...)` call allocates an
  independent, isolated in-memory database instance regardless of the URI suffix.

  ## Examples

      {:ok, db} = RuvectorElixir.open("my_db.rvf", 128)
      {:ok, db} = RuvectorElixir.open("my_db.rvf", 128, metric: :euclidean, hnsw: true)
      {:ok, mem_db} = RuvectorElixir.open("memory://transient", 64)

  """
  @spec open(Path.t(), pos_integer(), db_options() | keyword()) ::
          {:ok, db_ref()} | {:error, String.t()}
  def open(path, dimensions, options \\ %{})

  def open(path, dimensions, options)
      when is_binary(path) and byte_size(path) > 0 and is_integer(dimensions) and dimensions > 0 do
    if not String.valid?(path) do
      raise ArgumentError, "Database path must be a valid UTF-8 string"
    end

    Native.open_db_with_options(path, dimensions, normalize_options!(options, dimensions))
  end

  def open(path, dimensions, _options) do
    raise ArgumentError,
          "Invalid arguments to RuvectorElixir.open/3: expected non-empty binary path and positive integer dimensions, got path=#{inspect(path)}, dimensions=#{inspect(dimensions)}"
  end

  @doc """
  Opens a database, raising `RuvectorElixir.Error` on operational failure or `ArgumentError` on invalid arguments.
  """
  @spec open!(Path.t(), pos_integer(), db_options() | keyword()) :: db_ref()
  def open!(path, dimensions, options \\ %{}) do
    case open(path, dimensions, options) do
      {:ok, db} -> db
      {:error, reason} -> raise Error, reason
    end
  end

  @doc """
  Closes a database handle, releasing its underlying Rust database resources immediately.

  Idempotent: calling `close/1` on an already-closed handle returns `:ok`.
  Subsequent operations on a closed handle return `{:error, :closed}`.
  """
  @spec close(db_ref()) :: :ok | {:error, String.t()}
  def close(db_ref) when is_reference(db_ref), do: Native.close_db(db_ref)

  @doc """
  Closes a database handle, raising `RuvectorElixir.Error` on failure.
  """
  @spec close!(db_ref()) :: :ok
  def close!(db_ref) do
    case close(db_ref) do
      :ok -> :ok
      {:error, reason} -> raise Error, reason
    end
  end

  @doc """
  Opens a database, passes the handle to `fun`, and ensures `close/1` runs after `fun` completes or raises.

  Returns `{:ok, result}` on success, or `{:error, reason}` if `open/3` fails.
  """
  @spec with_open(Path.t(), pos_integer(), (db_ref() -> result)) ::
          {:ok, result} | {:error, String.t()}
        when result: var
  @spec with_open(Path.t(), pos_integer(), db_options() | keyword(), (db_ref() -> result)) ::
          {:ok, result} | {:error, String.t()}
        when result: var
  def with_open(path, dimensions, fun) when is_function(fun, 1) do
    with_open(path, dimensions, %{}, fun)
  end

  def with_open(_path, _dimensions, fun) do
    raise ArgumentError,
          "RuvectorElixir.with_open/3 expects a 1-arity function, got: #{inspect(fun)}"
  end

  def with_open(path, dimensions, opts, fun) when is_function(fun, 1) do
    case open(path, dimensions, opts) do
      {:ok, db} ->
        try do
          {:ok, fun.(db)}
        after
          close(db)
        end

      {:error, _} = error ->
        error
    end
  end

  def with_open(_path, _dimensions, _opts, fun) do
    raise ArgumentError,
          "RuvectorElixir.with_open/4 expects a 1-arity function, got: #{inspect(fun)}"
  end

  @doc """
  Opens a database with `open!/3`, passes the handle to `fun`, ensures `close/1` runs,
  and returns the result of `fun` directly.
  """
  @spec with_open!(Path.t(), pos_integer(), (db_ref() -> result)) :: result when result: var
  @spec with_open!(Path.t(), pos_integer(), db_options() | keyword(), (db_ref() -> result)) ::
          result
        when result: var
  def with_open!(path, dimensions, fun) when is_function(fun, 1) do
    with_open!(path, dimensions, %{}, fun)
  end

  def with_open!(_path, _dimensions, fun) do
    raise ArgumentError,
          "RuvectorElixir.with_open!/3 expects a 1-arity function, got: #{inspect(fun)}"
  end

  def with_open!(path, dimensions, opts, fun) when is_function(fun, 1) do
    db = open!(path, dimensions, opts)

    try do
      fun.(db)
    after
      close(db)
    end
  end

  def with_open!(_path, _dimensions, _opts, fun) do
    raise ArgumentError,
          "RuvectorElixir.with_open!/4 expects a 1-arity function, got: #{inspect(fun)}"
  end

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
  """
  @spec insert(db_ref(), entry_input()) :: {:ok, String.t()} | {:error, :closed | String.t()}
  def insert(db_ref, entry) when is_reference(db_ref), do: Native.insert(db_ref, entry)

  @doc """
  Inserts a vector entry into the database, raising `RuvectorElixir.Error` on failure.
  """
  @spec insert!(db_ref(), entry_input()) :: String.t()
  def insert!(db_ref, entry) do
    case insert(db_ref, entry) do
      {:ok, id} -> id
      {:error, reason} -> raise Error, reason
    end
  end

  @doc """
  Inserts a batch of vector entries into the database.

  Returns `{:ok, [id]}` on success, or `{:error, reason}` on failure.
  """
  @spec insert_batch(db_ref(), [entry_input()]) ::
          {:ok, [String.t()]} | {:error, :closed | String.t()}
  def insert_batch(db_ref, entries) when is_reference(db_ref) and is_list(entries) do
    Native.insert_batch(db_ref, entries)
  end

  @doc """
  Inserts a batch of vector entries into the database, raising `RuvectorElixir.Error` on failure.
  """
  @spec insert_batch!(db_ref(), [entry_input()]) :: [String.t()]
  def insert_batch!(db_ref, entries) do
    case insert_batch(db_ref, entries) do
      {:ok, ids} -> ids
      {:error, reason} -> raise Error, reason
    end
  end

  @doc """
  Retrieves a vector entry by its ID (string, non-boolean atom, or integer).

  Returns `{:ok, %{id: id, vector: vector, metadata: metadata}}` if found,
  `{:error, :not_found}` if no entry with that ID exists, `{:error, :closed}` if closed,
  or `{:error, reason}` on failure.
  """
  @spec get(db_ref(), id_input()) ::
          {:ok, entry_map()} | {:error, :not_found | :closed | String.t()}
  def get(db_ref, id) when is_reference(db_ref), do: Native.get_entry(db_ref, id)

  @doc """
  Retrieves a vector entry, raising `RuvectorElixir.Error` if not found or if an error occurs.
  """
  @spec get!(db_ref(), id_input()) :: entry_map()
  def get!(db_ref, id) do
    case get(db_ref, id) do
      {:ok, entry} -> entry
      {:error, reason} -> raise Error, reason
    end
  end

  @doc """
  Retrieves a vector entry and returns it as a `RuvectorElixir.VectorEntry` struct.
  """
  @spec get_struct(db_ref(), id_input()) ::
          {:ok, VectorEntry.t()} | {:error, :not_found | :closed | String.t()}
  def get_struct(db_ref, id) do
    case get(db_ref, id) do
      {:ok, entry_map} -> {:ok, VectorEntry.from_map(entry_map)}
      {:error, _} = error -> error
    end
  end

  @doc """
  Retrieves a vector entry as a `VectorEntry` struct, raising `RuvectorElixir.Error` on failure.
  """
  @spec get_struct!(db_ref(), id_input()) :: VectorEntry.t()
  def get_struct!(db_ref, id) do
    case get_struct(db_ref, id) do
      {:ok, struct} -> struct
      {:error, reason} -> raise Error, reason
    end
  end

  @doc """
  Deletes an entry by its ID (string, non-boolean atom, or integer).

  Returns `{:ok, true}` if deleted, `{:ok, false}` if not found, or `{:error, reason}` on failure.
  """
  @spec delete(db_ref(), id_input()) :: {:ok, boolean()} | {:error, :closed | String.t()}
  def delete(db_ref, id) when is_reference(db_ref), do: Native.delete(db_ref, id)

  @doc """
  Deletes an entry by its ID, raising `RuvectorElixir.Error` on failure.
  """
  @spec delete!(db_ref(), id_input()) :: boolean()
  def delete!(db_ref, id) do
    case delete(db_ref, id) do
      {:ok, deleted} -> deleted
      {:error, reason} -> raise Error, reason
    end
  end

  @doc """
  Returns the total number of vectors in the database.
  """
  @spec len(db_ref()) :: {:ok, non_neg_integer()} | {:error, :closed | String.t()}
  def len(db_ref) when is_reference(db_ref), do: Native.db_len(db_ref)

  @doc """
  Returns the total number of vectors in the database, raising `RuvectorElixir.Error` on error.
  """
  @spec len!(db_ref()) :: non_neg_integer()
  def len!(db_ref) do
    case len(db_ref) do
      {:ok, total} -> total
      {:error, reason} -> raise Error, reason
    end
  end

  @doc """
  Convenience alias for `len/1`.
  """
  @spec count(db_ref()) :: {:ok, non_neg_integer()} | {:error, :closed | String.t()}
  def count(db_ref), do: len(db_ref)

  @doc """
  Convenience alias for `len!/1`.
  """
  @spec count!(db_ref()) :: non_neg_integer()
  def count!(db_ref), do: len!(db_ref)

  @doc """
  Checks if the database is empty.

  Returns `true` or `false`, raising `RuvectorElixir.Error` on failure (such as `:closed`).
  """
  @spec empty?(db_ref()) :: boolean()
  def empty?(db_ref) when is_reference(db_ref) do
    case Native.is_empty(db_ref) do
      {:ok, empty} -> empty
      {:error, reason} -> raise Error, reason
    end
  end

  @doc """
  Returns a list of all vector IDs currently stored in the database.
  """
  @spec keys(db_ref()) :: {:ok, [String.t()]} | {:error, :closed | String.t()}
  def keys(db_ref) when is_reference(db_ref), do: Native.all_ids(db_ref)

  @doc """
  Returns all vector IDs in the database, raising `RuvectorElixir.Error` on error.
  """
  @spec keys!(db_ref()) :: [String.t()]
  def keys!(db_ref) do
    case keys(db_ref) do
      {:ok, ids} -> ids
      {:error, reason} -> raise Error, reason
    end
  end

  # ============================================================================
  # Search Operations
  # ============================================================================

  @doc """
  Searches the database for the top `k` closest vectors to `query_vector`.

  ## Options

  - `:filter` - Optional metadata filter map or keyword list (default: `nil`).
  - `:ef_search` - Optional positive integer HNSW `ef_search` override (default: `nil`).

  Returns `{:ok, [id]}` on success, or `{:error, reason}` on failure.
  """
  @spec search(db_ref(), vector(), non_neg_integer(), [search_option()] | map()) ::
          {:ok, [String.t()]} | {:error, :closed | String.t()}
  def search(db_ref, query_vector, k, opts \\ [])

  def search(db_ref, query_vector, k, opts)
      when is_reference(db_ref) and is_list(query_vector) and is_integer(k) and k >= 0 do
    %{filter: filter, ef_search: ef_search} =
      validate_search_opts!(opts, [:filter, :ef_search])

    Native.search(db_ref, query_vector, filter, k, ef_search)
  end

  def search(db_ref, query_vector, k, _opts) do
    raise ArgumentError,
          "Invalid arguments to RuvectorElixir.search/4: expected db_ref, list vector, and non-negative integer k, got db_ref=#{inspect(db_ref)}, query_vector=#{inspect(query_vector)}, k=#{inspect(k)}"
  end

  @doc """
  Searches the database for top `k` closest vectors, raising `RuvectorElixir.Error` on operational
  failure or `ArgumentError` on invalid arguments.
  """
  @spec search!(db_ref(), vector(), non_neg_integer(), [search_option()] | map()) :: [String.t()]
  def search!(db_ref, query_vector, k, opts \\ []) do
    case search(db_ref, query_vector, k, opts) do
      {:ok, ids} -> ids
      {:error, reason} -> raise Error, reason
    end
  end

  @doc """
  Searches the database for top `k` closest vectors, returning detailed search result maps.

  ## Options

  - `:filter` - Optional metadata filter map or keyword list (default: `nil`).
  - `:ef_search` - Optional positive integer HNSW `ef_search` override (default: `nil`).

  Each result map contains:
  - `:id` - Vector ID string
  - `:score` - Similarity / distance score (lower is closer for distance metrics)
  - `:vector` - Vector embedding float list (or `nil`)
  - `:metadata` - Metadata map (or `nil`)
  """
  @spec search_detailed(
          db_ref(),
          vector(),
          non_neg_integer(),
          [search_detailed_option()] | map()
        ) ::
          {:ok, [search_result()]} | {:error, :closed | String.t()}
  def search_detailed(db_ref, query_vector, k, opts \\ [])

  def search_detailed(db_ref, query_vector, k, opts)
      when is_reference(db_ref) and is_list(query_vector) and is_integer(k) and k >= 0 do
    %{filter: filter, ef_search: ef_search} =
      validate_search_opts!(opts, [:filter, :ef_search])

    Native.search_detailed(db_ref, query_vector, filter, k, ef_search)
  end

  def search_detailed(db_ref, query_vector, k, _opts) do
    raise ArgumentError,
          "Invalid arguments to RuvectorElixir.search_detailed/4: expected db_ref, list vector, and non-negative integer k, got db_ref=#{inspect(db_ref)}, query_vector=#{inspect(query_vector)}, k=#{inspect(k)}"
  end

  @doc """
  Searches the database returning detailed search results, raising `RuvectorElixir.Error` on
  operational failure or `ArgumentError` on invalid arguments.
  """
  @spec search_detailed!(
          db_ref(),
          vector(),
          non_neg_integer(),
          [search_detailed_option()] | map()
        ) ::
          [search_result()]
  def search_detailed!(db_ref, query_vector, k, opts \\ []) do
    case search_detailed(db_ref, query_vector, k, opts) do
      {:ok, results} -> results
      {:error, reason} -> raise Error, reason
    end
  end

  # ============================================================================
  # Distance Utilities & Info
  # ============================================================================

  @doc """
  Calculates the distance between two vectors `a` and `b` using the specified `metric`.

  Supported metrics: `:cosine`, `:euclidean`, `:dot_product`, `:dot`, `:manhattan`.
  Defaults to `:cosine`.

  Returns `{:ok, distance}` on success, or `{:error, reason}` on failure (e.g. dimension mismatch).
  """
  @spec distance(vector(), vector(), metric()) :: {:ok, float()} | {:error, String.t()}
  def distance(a, b, metric \\ :cosine)

  @dirty_distance_threshold 4096

  def distance(a, b, metric) when is_list(a) and a != [] and is_list(b) and b != [] do
    validated_metric = validate_metric!(metric)

    if length(a) > @dirty_distance_threshold or length(b) > @dirty_distance_threshold do
      Native.calculate_distance_dirty(a, b, validated_metric)
    else
      Native.calculate_distance(a, b, validated_metric)
    end
  end

  def distance(_a, _b, _metric) do
    raise ArgumentError, "RuvectorElixir.distance/3 expects two non-empty lists of numbers"
  end

  @doc """
  Calculates distance between two vectors, raising `RuvectorElixir.Error` on operational failure
  (e.g. dimension mismatch) or `ArgumentError` on invalid inputs.
  """
  @spec distance!(vector(), vector(), metric()) :: float()
  def distance!(a, b, metric \\ :cosine) do
    case distance(a, b, metric) do
      {:ok, dist} -> dist
      {:error, reason} -> raise Error, reason
    end
  end

  @doc """
  Gets database metadata info including `:dimensions`, `:node_count`, `:distance_metric`,
  `:storage_path`, `:hnsw`, and `:quantization`.
  """
  @spec info(db_ref()) :: {:ok, db_info()} | {:error, :closed | String.t()}
  def info(db_ref) when is_reference(db_ref), do: Native.info(db_ref)

  @doc """
  Gets database metadata info, raising `RuvectorElixir.Error` on failure.
  """
  @spec info!(db_ref()) :: db_info()
  def info!(db_ref) do
    case info(db_ref) do
      {:ok, info_map} -> info_map
      {:error, reason} -> raise Error, reason
    end
  end

  # ============================================================================
  # Option & Argument Validation Helpers
  # ============================================================================

  defp normalize_options!(opts, dimensions) when is_list(opts) do
    if not Keyword.keyword?(opts) do
      raise ArgumentError, "Database options must be a keyword list or plain map"
    end

    reduce_db_options!(opts, dimensions)
  end

  defp normalize_options!(%{} = map, dimensions) when not is_struct(map) do
    reduce_db_options!(map, dimensions)
  end

  defp normalize_options!(other, _dimensions) do
    raise ArgumentError,
          "Database options must be a keyword list or plain map, got: #{inspect(other)}"
  end

  defp reduce_db_options!(enum, dimensions) do
    Enum.reduce(enum, %{}, fn {k, v}, acc ->
      key = normalize_opt_key!(k)

      if Map.has_key?(acc, key) do
        raise ArgumentError, "Duplicate database option: #{inspect(key)}"
      end

      if key in ["metric", "distance_metric"] and
           (Map.has_key?(acc, "metric") or Map.has_key?(acc, "distance_metric")) do
        raise ArgumentError, "Cannot specify both :metric and :distance_metric"
      end

      val =
        case key do
          "metric" ->
            validate_metric!(v)

          "distance_metric" ->
            validate_metric!(v)

          "hnsw" ->
            normalize_hnsw!(v)

          "quantization" ->
            normalize_quantization!(v, dimensions)

          _ ->
            raise ArgumentError, "Unknown database option: #{inspect(k)}"
        end

      Map.put(acc, key, val)
    end)
  end

  defp normalize_opt_key!(k) when is_boolean(k) or is_nil(k) do
    raise ArgumentError, "Invalid option key: #{inspect(k)}"
  end

  defp normalize_opt_key!(k) when is_atom(k), do: Atom.to_string(k)
  defp normalize_opt_key!(k) when is_binary(k), do: k

  defp normalize_opt_key!(k) do
    raise ArgumentError, "Invalid option key: #{inspect(k)}"
  end

  defp validate_metric!(metric) when metric in @valid_metrics, do: metric

  defp validate_metric!(metric) when is_binary(metric) do
    if String.downcase(metric) in @valid_metric_strings do
      metric
    else
      raise ArgumentError, "Unsupported distance metric: #{inspect(metric)}"
    end
  end

  defp validate_metric!(other) do
    raise ArgumentError, "Unsupported distance metric: #{inspect(other)}"
  end

  defp normalize_hnsw!(nil), do: nil
  defp normalize_hnsw!(b) when is_boolean(b), do: b

  defp normalize_hnsw!([]), do: raise(ArgumentError, ":hnsw option list cannot be empty")

  defp normalize_hnsw!(opts) when is_list(opts) do
    if not Keyword.keyword?(opts) do
      raise ArgumentError, ":hnsw options must be a keyword list, plain map, or boolean"
    end

    reduce_hnsw_options!(opts)
  end

  defp normalize_hnsw!(%{} = map) when not is_struct(map) do
    if map_size(map) == 0 do
      raise ArgumentError, ":hnsw option map cannot be empty"
    end

    reduce_hnsw_options!(map)
  end

  defp normalize_hnsw!(other) do
    raise ArgumentError,
          ":hnsw must be a boolean, keyword list, or plain map, got: #{inspect(other)}"
  end

  defp reduce_hnsw_options!(enum) do
    Enum.reduce(enum, %{}, fn {hk, hv}, acc ->
      hkey = normalize_opt_key!(hk)

      if Map.has_key?(acc, hkey) do
        raise ArgumentError, "Duplicate :hnsw option: #{inspect(hkey)}"
      end

      if hkey not in ["m", "ef_construction", "ef_search", "max_elements"] do
        raise ArgumentError, "Unknown :hnsw option: #{inspect(hk)}"
      end

      if not (is_integer(hv) and hv > 0) do
        raise ArgumentError,
              ":hnsw option :#{hkey} must be a positive integer, got: #{inspect(hv)}"
      end

      if hkey == "m" and hv not in 2..256 do
        raise ArgumentError, ":hnsw option :m must be between 2 and 256, got: #{inspect(hv)}"
      end

      Map.put(acc, hkey, hv)
    end)
  end

  defp normalize_quantization!(nil, _dimensions), do: nil

  defp normalize_quantization!(q, _dimensions) when q in [:scalar, :binary, :none], do: q

  defp normalize_quantization!(q, _dimensions) when is_binary(q) do
    if String.downcase(q) in ["scalar", "binary", "none"] do
      q
    else
      raise ArgumentError, "Unsupported :quantization option: #{inspect(q)}"
    end
  end

  defp normalize_quantization!([], _dimensions) do
    raise ArgumentError, ":quantization option list cannot be empty"
  end

  defp normalize_quantization!(opts, dimensions) when is_list(opts) do
    if not Keyword.keyword?(opts) do
      raise ArgumentError,
            ":quantization options must be a keyword list, plain map, or atom/string"
    end

    reduce_quantization_options!(opts, dimensions)
  end

  defp normalize_quantization!(%{} = map, dimensions) when not is_struct(map) do
    if map_size(map) == 0 do
      raise ArgumentError, ":quantization option map cannot be empty"
    end

    reduce_quantization_options!(map, dimensions)
  end

  defp normalize_quantization!(other, _dimensions) do
    raise ArgumentError, "Unsupported :quantization option: #{inspect(other)}"
  end

  defp reduce_quantization_options!(enum, dimensions) do
    normalized =
      Enum.reduce(enum, %{}, fn {qk, qv}, acc ->
        qkey = normalize_opt_key!(qk)

        if Map.has_key?(acc, qkey) do
          raise ArgumentError, "Duplicate :quantization option: #{inspect(qkey)}"
        end

        val =
          case qkey do
            "type" ->
              validate_quantization_type!(qv)

            param when param in ["subspaces", "k"] ->
              if not (is_integer(qv) and qv > 0) do
                raise ArgumentError,
                      ":quantization :#{param} must be a positive integer, got: #{inspect(qv)}"
              end

              qv

            _ ->
              raise ArgumentError, "Unknown :quantization option: #{inspect(qk)}"
          end

        Map.put(acc, qkey, val)
      end)

    type_val = Map.get(normalized, "type")
    normalized_type = normalize_quantization_type_name(type_val)
    has_product_params = Map.has_key?(normalized, "subspaces") or Map.has_key?(normalized, "k")

    if normalized_type in ["scalar", "binary", "none"] and has_product_params do
      raise ArgumentError,
            ":subspaces and :k are only valid with :product quantization, not #{inspect(type_val)}"
    end

    is_product = normalized_type == "product" or (is_nil(normalized_type) and has_product_params)

    if is_product do
      subspaces = Map.get(normalized, "subspaces", 8)

      if rem(dimensions, subspaces) != 0 do
        raise ArgumentError,
              ":quantization :subspaces (#{subspaces}) must evenly divide dimensions (#{dimensions})"
      end
    end

    normalized
  end

  defp normalize_quantization_type_name(nil), do: nil
  defp normalize_quantization_type_name(t) when is_atom(t), do: Atom.to_string(t)
  defp normalize_quantization_type_name(t) when is_binary(t), do: String.downcase(t)

  defp validate_quantization_type!(t) when t in [:scalar, :binary, :none, :product], do: t

  defp validate_quantization_type!(t) when is_binary(t) do
    if String.downcase(t) in ["scalar", "binary", "none", "product"] do
      t
    else
      raise ArgumentError, "Unsupported :quantization :type: #{inspect(t)}"
    end
  end

  defp validate_quantization_type!(other) do
    raise ArgumentError, "Unsupported :quantization :type: #{inspect(other)}"
  end

  defp validate_search_opts!(opts, allowed_keys) when is_list(opts) do
    if not Keyword.keyword?(opts) do
      raise ArgumentError, "Search options must be a keyword list or plain map"
    end

    reduce_search_opts!(opts, allowed_keys)
  end

  defp validate_search_opts!(%{} = opts, allowed_keys) when not is_struct(opts) do
    reduce_search_opts!(opts, allowed_keys)
  end

  defp validate_search_opts!(other, _allowed_keys) do
    raise ArgumentError,
          "Search options must be a keyword list or plain map, got: #{inspect(other)}"
  end

  defp reduce_search_opts!(enum, allowed_keys) do
    acc =
      Enum.reduce(enum, %{seen: MapSet.new(), filter: nil, ef_search: nil}, fn {k, v}, acc ->
        if k not in allowed_keys do
          raise ArgumentError,
                "Unknown search option #{inspect(k)}; allowed options: #{inspect(allowed_keys)}"
        end

        if MapSet.member?(acc.seen, k) do
          raise ArgumentError, "Duplicate search option: #{inspect(k)}"
        end

        case k do
          :filter ->
            validate_filter!(v)

          :ef_search ->
            if v != nil and not (is_integer(v) and v > 0) do
              raise ArgumentError,
                    ":ef_search must be a positive integer or nil, got: #{inspect(v)}"
            end
        end

        acc
        |> Map.put(k, v)
        |> Map.update!(:seen, &MapSet.put(&1, k))
      end)

    %{filter: acc.filter, ef_search: acc.ef_search}
  end

  defp validate_filter!(nil), do: :ok

  defp validate_filter!(%{} = filter)
       when not is_struct(filter) and not is_map_key(filter, :__struct__),
       do: :ok

  defp validate_filter!(filter) when is_list(filter) do
    if valid_filter_list?(filter) do
      :ok
    else
      raise ArgumentError,
            ":filter list must be a keyword list or list of {string_or_atom, value} pairs, got: #{inspect(filter)}"
    end
  end

  defp validate_filter!(other) do
    raise ArgumentError,
          ":filter must be a plain map, keyword list, or nil, got: #{inspect(other)}"
  end

  defp valid_filter_list?([]), do: true

  defp valid_filter_list?([{k, _v} | rest]) when is_binary(k) do
    String.valid?(k) and valid_filter_list?(rest)
  end

  defp valid_filter_list?([{k, _v} | rest])
       when is_atom(k) and not is_boolean(k) and not is_nil(k) and k != :__struct__ do
    valid_filter_list?(rest)
  end

  defp valid_filter_list?(_other), do: false
end
