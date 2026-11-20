defmodule RuvectorElixir.VectorEntry do
  @moduledoc """
  Represents a vector entry stored in or retrieved from a Ruvector database.

  An entry consists of:
  - `id`: Optional unique string identifier (auto-generated UUID if omitted upon insert).
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

  Accepts vectors with float or integer components (automatically converted to floats).

  ## Options

  - `:id` - Unique string identifier (or non-boolean atom / integer).
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

    validate_option_keys!(Keyword.keys(opts))

    %__MODULE__{
      vector: normalize_vector!(vector),
      id: normalize_id!(Keyword.get(opts, :id)),
      metadata: normalize_metadata!(Keyword.get(opts, :metadata))
    }
  end

  def new(vector, %{} = opts) when is_list(vector) and not is_struct(opts) do
    validate_option_keys!(Map.keys(opts))

    raw_id = fetch_key(opts, :id, "id")
    raw_metadata = fetch_key(opts, :metadata, "metadata")

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
  def from_map(%{vector: vector} = map) when is_list(vector) do
    %__MODULE__{
      vector: ensure_float_list!(vector),
      id: normalize_id!(fetch_key(map, :id, "id")),
      metadata: normalize_metadata!(fetch_key(map, :metadata, "metadata"))
    }
  end

  def from_map(%{"vector" => vector} = map) when is_list(vector) do
    %__MODULE__{
      vector: ensure_float_list!(vector),
      id: normalize_id!(fetch_key(map, :id, "id")),
      metadata: normalize_metadata!(fetch_key(map, :metadata, "metadata"))
    }
  end

  def from_map(_other) do
    raise ArgumentError, "VectorEntry.from_map/1 expects a map with a :vector list"
  end

  defp fetch_key(map, atom_key, string_key) do
    case Map.fetch(map, atom_key) do
      {:ok, value} -> value
      :error -> Map.get(map, string_key)
    end
  end

  defp validate_option_keys!(keys) do
    Enum.each(keys, fn key ->
      if key not in [:id, :metadata, "id", "metadata"] do
        raise ArgumentError, "Unknown VectorEntry option: #{inspect(key)}"
      end
    end)
  end

  defp normalize_id!(nil), do: nil
  defp normalize_id!(bool) when is_boolean(bool), do: raise(ArgumentError, "ID cannot be boolean")
  defp normalize_id!(atom) when is_atom(atom), do: Atom.to_string(atom)
  defp normalize_id!(int) when is_integer(int), do: Integer.to_string(int)
  defp normalize_id!(""), do: raise(ArgumentError, "VectorEntry :id cannot be an empty string")
  defp normalize_id!(str) when is_binary(str), do: str

  defp normalize_id!(other) do
    raise ArgumentError,
          "VectorEntry :id must be a string, non-boolean atom, integer, or nil, got: #{inspect(other)}"
  end

  defp normalize_vector!(vector) do
    Enum.map(vector, fn
      val when is_float(val) -> val
      val when is_integer(val) -> val * 1.0
      other -> raise ArgumentError, "Vector elements must be numbers, got: #{inspect(other)}"
    end)
  end

  defp ensure_float_list!(vector) do
    if Enum.all?(vector, &is_float/1) do
      vector
    else
      normalize_vector!(vector)
    end
  end

  defp normalize_metadata!(nil), do: nil

  defp normalize_metadata!(%{} = metadata) when not is_struct(metadata), do: metadata

  defp normalize_metadata!(other) do
    raise ArgumentError,
          "VectorEntry :metadata must be a plain map or nil, got: #{inspect(other)}"
  end
end
