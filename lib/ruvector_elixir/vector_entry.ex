defmodule RuvectorElixir.VectorEntry do
  @moduledoc """
  Represents a vector entry stored in or retrieved from a Ruvector database.

  An entry consists of:
  - `id`: Optional unique UTF-8 string identifier (auto-generated UUID if omitted upon insert).
  - `vector`: List of floating point numbers representing the vector embedding (stored as 32-bit `f32` in Rust; non-dyadic 64-bit floats are rounded to `f32` precision on insert).
  - `metadata`: Optional arbitrary map of JSON-serializable key-value metadata.
  """

  @enforce_keys [:vector]
  defstruct [:id, :vector, metadata: nil]

  @type t :: %__MODULE__{
          id: String.t() | nil,
          vector: [float()],
          metadata: map() | nil
        }

  @doc """
  Creates a new `VectorEntry` struct.

  Accepts vectors with finite `f32`-representable float or integer components (automatically
  converted to floats).

  ## Options

  - `:id` - Unique UTF-8 string identifier (or non-boolean atom / integer).
  - `:metadata` - Plain map of JSON-serializable metadata attributes.

  ## Examples

      iex> RuvectorElixir.VectorEntry.new([1.0, 2.0, 3.0])
      %RuvectorElixir.VectorEntry{id: nil, vector: [1.0, 2.0, 3.0], metadata: nil}

      iex> RuvectorElixir.VectorEntry.new([1, 2, 3], id: "vec_1", metadata: %{"label" => "sample"})
      %RuvectorElixir.VectorEntry{id: "vec_1", vector: [1.0, 2.0, 3.0], metadata: %{"label" => "sample"}}

  """
  @spec new([number()], keyword() | map()) :: t()
  def new(vector, opts \\ [])

  def new(vector, opts) when is_list(vector) and is_list(opts) do
    if not Keyword.keyword?(opts) do
      raise ArgumentError, "VectorEntry options must be a keyword list or map"
    end

    %{id: raw_id, metadata: raw_metadata} = extract_opts!(opts)

    %__MODULE__{
      vector: normalize_vector!(vector),
      id: normalize_id!(raw_id),
      metadata: normalize_metadata!(raw_metadata)
    }
  end

  def new(vector, %{} = opts) when is_list(vector) and not is_struct(opts) do
    %{id: raw_id, metadata: raw_metadata} = extract_opts!(opts)

    %__MODULE__{
      vector: normalize_vector!(vector),
      id: normalize_id!(raw_id),
      metadata: normalize_metadata!(raw_metadata)
    }
  end

  def new(_vector, _opts) do
    raise ArgumentError,
          "VectorEntry.new/2 expects a list of numbers and a keyword list or plain map of options"
  end

  @doc """
  Converts a map returned by `RuvectorElixir.get/2` into a `VectorEntry` struct.

  ## Examples

      iex> map = %{id: "doc_1", vector: [0.1, 0.2], metadata: %{"tag" => "A"}}
      iex> RuvectorElixir.VectorEntry.from_map(map)
      %RuvectorElixir.VectorEntry{id: "doc_1", vector: [0.1, 0.2], metadata: %{"tag" => "A"}}

  """
  @spec from_map(map()) :: t()
  def from_map(%{} = map) when not is_struct(map) do
    if Map.has_key?(map, :vector) and Map.has_key?(map, "vector") do
      raise ArgumentError, "VectorEntry.from_map/1 map cannot contain both :vector and \"vector\""
    end

    vector =
      case Map.fetch(map, :vector) do
        {:ok, v} -> v
        :error -> Map.get(map, "vector")
      end

    if not is_list(vector) do
      raise ArgumentError, "VectorEntry.from_map/1 expects a map with a :vector list"
    end

    %{id: raw_id, metadata: raw_metadata} = extract_entry_fields!(map)

    %__MODULE__{
      vector: ensure_float_list!(vector),
      id: normalize_id!(raw_id),
      metadata: normalize_metadata!(raw_metadata)
    }
  end

  def from_map(_other) do
    raise ArgumentError, "VectorEntry.from_map/1 expects a plain map with a :vector list"
  end

  defp extract_opts!(enum) do
    Enum.reduce(enum, %{id: nil, metadata: nil, seen: MapSet.new()}, fn {k, v}, acc ->
      key =
        case k do
          k when k in [:id, "id"] -> :id
          k when k in [:metadata, "metadata"] -> :metadata
          other -> raise ArgumentError, "Unknown VectorEntry option: #{inspect(other)}"
        end

      if MapSet.member?(acc.seen, key) do
        raise ArgumentError, "Duplicate VectorEntry option: #{inspect(key)}"
      end

      acc
      |> Map.put(key, v)
      |> Map.update!(:seen, &MapSet.put(&1, key))
    end)
  end

  defp extract_entry_fields!(map) do
    if Map.has_key?(map, :id) and Map.has_key?(map, "id") do
      raise ArgumentError, "VectorEntry.from_map/1 map cannot contain both :id and \"id\""
    end

    if Map.has_key?(map, :metadata) and Map.has_key?(map, "metadata") do
      raise ArgumentError,
            "VectorEntry.from_map/1 map cannot contain both :metadata and \"metadata\""
    end

    id =
      case Map.fetch(map, :id) do
        {:ok, v} -> v
        :error -> Map.get(map, "id")
      end

    metadata =
      case Map.fetch(map, :metadata) do
        {:ok, v} -> v
        :error -> Map.get(map, "metadata")
      end

    %{id: id, metadata: metadata}
  end

  defp normalize_id!(nil), do: nil
  defp normalize_id!(bool) when is_boolean(bool), do: raise(ArgumentError, "ID cannot be boolean")
  defp normalize_id!(atom) when is_atom(atom), do: Atom.to_string(atom)
  defp normalize_id!(int) when is_integer(int), do: Integer.to_string(int)
  defp normalize_id!(""), do: raise(ArgumentError, "VectorEntry :id cannot be an empty string")

  defp normalize_id!(str) when is_binary(str) do
    if not String.valid?(str) do
      raise ArgumentError, "VectorEntry :id must be a valid UTF-8 string"
    end

    str
  end

  defp normalize_id!(other) do
    raise ArgumentError,
          "VectorEntry :id must be a string, non-boolean atom, integer, or nil, got: #{inspect(other)}"
  end

  defp normalize_vector!(vector) do
    Enum.map(vector, &to_finite_f32_float!/1)
  end

  defp ensure_float_list!(vector) do
    if Enum.all?(vector, &(is_float(&1) and finite_f32?(&1))) do
      vector
    else
      normalize_vector!(vector)
    end
  end

  defp to_finite_f32_float!(val) when is_float(val) do
    if finite_f32?(val) do
      val
    else
      raise ArgumentError, "Vector element overflows 32-bit float range: #{inspect(val)}"
    end
  end

  defp to_finite_f32_float!(val) when is_integer(val) do
    f = val * 1.0

    if finite_f32?(f) do
      f
    else
      raise ArgumentError, "Vector element overflows 32-bit float range: #{inspect(val)}"
    end
  end

  defp to_finite_f32_float!(other) do
    raise ArgumentError, "Vector elements must be numbers, got: #{inspect(other)}"
  end

  defp finite_f32?(val) do
    not match?(<<_::1, 255::8, _::23>>, <<val::float-32>>)
  end

  defp normalize_metadata!(nil), do: nil

  defp normalize_metadata!(%{} = metadata) when not is_struct(metadata), do: metadata

  defp normalize_metadata!(other) do
    raise ArgumentError,
          "VectorEntry :metadata must be a plain map or nil, got: #{inspect(other)}"
  end
end
