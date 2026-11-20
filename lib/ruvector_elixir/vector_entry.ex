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
          metadata: %{optional(String.t()) => term()} | nil
        }

  @doc """
  Creates a new `VectorEntry` struct.

  Accepts vectors with finite `f32`-representable float or integer components (automatically
  converted to floats).

  ## Options

  - `:id` - Unique UTF-8 string identifier (or non-boolean atom / integer).
  - `:metadata` - Plain map of JSON-serializable metadata attributes. Atom keys and
    non-boolean/non-`nil` atom values are canonicalized to strings so `RuvectorElixir.get_struct/2`
    round-trips identically.

  ## Examples

      iex> RuvectorElixir.VectorEntry.new([1.0, 2.0, 3.0])
      %RuvectorElixir.VectorEntry{id: nil, vector: [1.0, 2.0, 3.0], metadata: nil}

      iex> RuvectorElixir.VectorEntry.new([1, 2, 3], id: "vec_1", metadata: %{"label" => "sample"})
      %RuvectorElixir.VectorEntry{id: "vec_1", vector: [1.0, 2.0, 3.0], metadata: %{"label" => "sample"}}

  """
  @spec new([number()], keyword() | map()) :: t()
  def new(vector, opts \\ [])

  def new(vector, opts) when is_list(vector) and vector != [] and is_list(opts) do
    if not Keyword.keyword?(opts) do
      raise ArgumentError, "VectorEntry options must be a keyword list or map"
    end

    %{id: raw_id, metadata: raw_metadata} = extract_opts!(opts)

    %__MODULE__{
      vector: ensure_float_list!(vector),
      id: normalize_id!(raw_id),
      metadata: normalize_metadata!(raw_metadata)
    }
  end

  def new(vector, %{} = opts)
      when is_list(vector) and vector != [] and not is_struct(opts) do
    %{id: raw_id, metadata: raw_metadata} = extract_opts!(opts)

    %__MODULE__{
      vector: ensure_float_list!(vector),
      id: normalize_id!(raw_id),
      metadata: normalize_metadata!(raw_metadata)
    }
  end

  def new(_vector, _opts) do
    raise ArgumentError,
          "VectorEntry.new/2 expects a non-empty list of numbers and a keyword list or plain map of options"
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

    if not is_list(vector) or vector == [] do
      raise ArgumentError, "VectorEntry.from_map/1 expects a map with a non-empty :vector list"
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

  @min_i128 -170_141_183_460_469_231_731_687_303_715_884_105_728
  @max_u128 340_282_366_920_938_463_463_374_607_431_768_211_455

  defp normalize_id!(nil), do: nil
  defp normalize_id!(bool) when is_boolean(bool), do: raise(ArgumentError, "ID cannot be boolean")

  defp normalize_id!(:"") do
    raise ArgumentError, "VectorEntry :id cannot be an empty atom"
  end

  defp normalize_id!(atom) when is_atom(atom), do: Atom.to_string(atom)

  defp normalize_id!(int) when is_integer(int) and (int < @min_i128 or int > @max_u128) do
    raise ArgumentError,
          "VectorEntry :id integer out of 128-bit range: #{inspect(int)}"
  end

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
    do_normalize_vector!(vector, [])
  end

  defp do_normalize_vector!([], acc), do: Enum.reverse(acc)

  defp do_normalize_vector!([head | tail], acc) do
    do_normalize_vector!(tail, [to_finite_f32_float!(head) | acc])
  end

  defp do_normalize_vector!(_improper_tail, _acc) do
    raise ArgumentError, "Vector must be a proper list of numbers"
  end

  defp ensure_float_list!(vector) do
    if all_finite_f32_floats?(vector) do
      vector
    else
      normalize_vector!(vector)
    end
  end

  defp all_finite_f32_floats?([]), do: true

  defp all_finite_f32_floats?([head | tail]) when is_float(head) do
    finite_f32?(head) and all_finite_f32_floats?(tail)
  end

  defp all_finite_f32_floats?(_other), do: false

  # 2^128: integers at or above this magnitude overflow IEEE-754 f32 before f64 conversion.
  @f32_integer_overflow_bound 0x1_0000_0000_0000_0000_0000_0000_0000_0000

  defp to_finite_f32_float!(val) when is_float(val) do
    ensure_finite_f32!(val, val)
  end

  defp to_finite_f32_float!(val)
       when is_integer(val) and abs(val) >= @f32_integer_overflow_bound do
    raise_f32_overflow!(val)
  end

  defp to_finite_f32_float!(val) when is_integer(val) do
    ensure_finite_f32!(val * 1.0, val)
  end

  defp to_finite_f32_float!(other) do
    raise ArgumentError, "Vector elements must be numbers, got: #{inspect(other)}"
  end

  defp ensure_finite_f32!(float_val, original_val) do
    if finite_f32?(float_val) do
      float_val
    else
      raise_f32_overflow!(original_val)
    end
  end

  defp raise_f32_overflow!(val) do
    raise ArgumentError, "Vector element overflows 32-bit float range: #{inspect(val)}"
  end

  defp finite_f32?(val) do
    not match?(<<_::1, 255::8, _::23>>, <<val::float-32>>)
  end

  @max_json_depth 127
  @min_i64 -9_223_372_036_854_775_808
  @max_u64 18_446_744_073_709_551_615

  defp normalize_metadata!(nil), do: nil

  defp normalize_metadata!(%{} = metadata)
       when not is_struct(metadata) and not is_map_key(metadata, :__struct__) do
    normalize_json_map!(metadata, 1)
  end

  defp normalize_metadata!(other) do
    raise ArgumentError,
          "VectorEntry :metadata must be a plain map or nil, got: #{inspect(other)}"
  end

  defp normalize_json_map!(map, depth) do
    if depth > @max_json_depth do
      raise ArgumentError,
            "VectorEntry :metadata exceeds maximum nesting depth of #{@max_json_depth}"
    end

    Enum.reduce(map, %{}, fn {k, v}, acc ->
      norm_k = normalize_metadata_key!(k)

      if Map.has_key?(acc, norm_k) do
        raise ArgumentError, "Duplicate VectorEntry :metadata key: #{inspect(norm_k)}"
      end

      Map.put(acc, norm_k, normalize_json_value!(v, depth))
    end)
  end

  defp normalize_metadata_key!(k) when is_binary(k) do
    if not String.valid?(k) do
      raise ArgumentError, "VectorEntry :metadata keys must be valid UTF-8 strings"
    end

    k
  end

  defp normalize_metadata_key!(k)
       when is_atom(k) and not is_boolean(k) and not is_nil(k) and k != :__struct__ do
    Atom.to_string(k)
  end

  defp normalize_metadata_key!(other) do
    raise ArgumentError,
          "VectorEntry :metadata keys must be valid UTF-8 strings or non-boolean atoms, got: #{inspect(other)}"
  end

  defp normalize_json_value!(nil, _depth), do: nil
  defp normalize_json_value!(val, _depth) when is_boolean(val), do: val
  defp normalize_json_value!(val, _depth) when is_float(val), do: val

  defp normalize_json_value!(val, _depth)
       when is_integer(val) and val >= @min_i64 and val <= @max_u64,
       do: val

  defp normalize_json_value!(val, _depth) when is_integer(val) do
    raise ArgumentError,
          "VectorEntry :metadata integer out of 64-bit JSON range: #{inspect(val)}"
  end

  defp normalize_json_value!(val, _depth) when is_binary(val) do
    if not String.valid?(val) do
      raise ArgumentError, "VectorEntry :metadata strings must be valid UTF-8"
    end

    val
  end

  defp normalize_json_value!(val, _depth) when is_atom(val), do: Atom.to_string(val)

  defp normalize_json_value!(val, depth) when is_list(val) do
    next_depth = depth + 1

    if next_depth > @max_json_depth do
      raise ArgumentError,
            "VectorEntry :metadata exceeds maximum nesting depth of #{@max_json_depth}"
    end

    normalize_json_list!(val, next_depth, [])
  end

  defp normalize_json_value!(%{} = map, depth)
       when not is_struct(map) and not is_map_key(map, :__struct__) do
    normalize_json_map!(map, depth + 1)
  end

  defp normalize_json_value!(other, _depth) do
    raise ArgumentError,
          "VectorEntry :metadata values must be JSON-compatible terms, got: #{inspect(other)}"
  end

  defp normalize_json_list!([], _item_depth, acc), do: Enum.reverse(acc)

  defp normalize_json_list!([head | tail], item_depth, acc) do
    normalize_json_list!(tail, item_depth, [normalize_json_value!(head, item_depth) | acc])
  end

  defp normalize_json_list!(_improper_tail, _item_depth, _acc) do
    raise ArgumentError, "VectorEntry :metadata lists must be proper lists"
  end
end
