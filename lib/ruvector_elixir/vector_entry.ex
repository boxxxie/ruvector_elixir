defmodule RuvectorElixir.VectorEntry do
  @moduledoc """
  Represents a vector entry stored in or retrieved from a Ruvector database.

  An entry consists of:
  - `id`: Optional unique string identifier (auto-generated UUID if omitted upon insert).
  - `vector`: List of 32-bit floating point numbers representing the vector embedding.
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

  - `:id` - Unique string identifier (or atom / integer).
  - `:metadata` - Map of metadata attributes.

  ## Examples

      iex> RuvectorElixir.VectorEntry.new([1.0, 2.0, 3.0])
      %RuvectorElixir.VectorEntry{id: nil, vector: [1.0, 2.0, 3.0], metadata: nil}

      iex> RuvectorElixir.VectorEntry.new([1, 2, 3], id: "vec_1", metadata: %{"label" => "sample"})
      %RuvectorElixir.VectorEntry{id: "vec_1", vector: [1.0, 2.0, 3.0], metadata: %{"label" => "sample"}}

  """
  @spec new([number()], keyword() | map()) :: t()
  def new(vector, opts \\ [])

  def new(vector, opts) when is_list(vector) and is_list(opts) do
    id =
      case Keyword.get(opts, :id) do
        nil -> nil
        atom when is_atom(atom) -> Atom.to_string(atom)
        int when is_integer(int) -> Integer.to_string(int)
        str when is_binary(str) -> str
        other -> to_string(other)
      end

    %__MODULE__{
      vector: Enum.map(vector, &(&1 * 1.0)),
      id: id,
      metadata: Keyword.get(opts, :metadata)
    }
  end

  def new(vector, %{} = opts) when is_list(vector) do
    raw_id = Map.get(opts, :id) || Map.get(opts, "id")

    id =
      case raw_id do
        nil -> nil
        atom when is_atom(atom) -> Atom.to_string(atom)
        int when is_integer(int) -> Integer.to_string(int)
        str when is_binary(str) -> str
        other -> to_string(other)
      end

    %__MODULE__{
      vector: Enum.map(vector, &(&1 * 1.0)),
      id: id,
      metadata: Map.get(opts, :metadata) || Map.get(opts, "metadata")
    }
  end

  @doc """
  Converts a map returned by `RuvectorElixir.get/2` or `RuvectorElixir.get_entry/2` into a `VectorEntry` struct.

  ## Examples

      iex> map = %{id: "doc_1", vector: [0.1, 0.2], metadata: %{"tag" => "A"}}
      iex> RuvectorElixir.VectorEntry.from_map(map)
      %RuvectorElixir.VectorEntry{id: "doc_1", vector: [0.1, 0.2], metadata: %{"tag" => "A"}}

  """
  @spec from_map(map()) :: t()
  def from_map(%{vector: vector} = map) do
    %__MODULE__{
      vector: Enum.map(vector, &(&1 * 1.0)),
      id: Map.get(map, :id) || Map.get(map, "id"),
      metadata: Map.get(map, :metadata) || Map.get(map, "metadata")
    }
  end

  def from_map(%{"vector" => vector} = map) do
    %__MODULE__{
      vector: Enum.map(vector, &(&1 * 1.0)),
      id: Map.get(map, :id) || Map.get(map, "id"),
      metadata: Map.get(map, :metadata) || Map.get(map, "metadata")
    }
  end
end
