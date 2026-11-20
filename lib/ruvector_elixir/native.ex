defmodule RuvectorElixir.Native do
  @moduledoc false

  use Rustler, otp_app: :ruvector_elixir, crate: "ruvector_nif"

  @spec open_db_with_options(String.t(), pos_integer(), %{optional(String.t()) => term()} | nil) ::
          {:ok, reference()} | {:error, String.t()}
  def open_db_with_options(_path, _dimensions, _options), do: :erlang.nif_error(:nif_not_loaded)

  @spec close_db(reference()) :: :ok | {:error, String.t()}
  def close_db(_db_ref), do: :erlang.nif_error(:nif_not_loaded)

  @spec insert(reference(), term()) :: {:ok, String.t()} | {:error, :closed | String.t()}
  def insert(_db_ref, _entry), do: :erlang.nif_error(:nif_not_loaded)

  @spec insert_batch(reference(), term()) ::
          {:ok, [String.t()]} | {:error, :closed | String.t()}
  def insert_batch(_db_ref, _entries), do: :erlang.nif_error(:nif_not_loaded)

  @spec get_entry(reference(), term()) ::
          {:ok, map()} | {:error, :not_found | :closed | String.t()}
  def get_entry(_db_ref, _id), do: :erlang.nif_error(:nif_not_loaded)

  @spec delete(reference(), term()) :: {:ok, boolean()} | {:error, :closed | String.t()}
  def delete(_db_ref, _id), do: :erlang.nif_error(:nif_not_loaded)

  @spec db_len(reference()) :: {:ok, non_neg_integer()} | {:error, :closed | String.t()}
  def db_len(_db_ref), do: :erlang.nif_error(:nif_not_loaded)

  @spec is_empty(reference()) :: {:ok, boolean()} | {:error, :closed | String.t()}
  def is_empty(_db_ref), do: :erlang.nif_error(:nif_not_loaded)

  @spec all_ids(reference()) :: {:ok, [String.t()]} | {:error, :closed | String.t()}
  def all_ids(_db_ref), do: :erlang.nif_error(:nif_not_loaded)

  @spec search(
          reference(),
          [number()],
          map() | keyword() | nil,
          non_neg_integer(),
          pos_integer() | nil
        ) ::
          {:ok, [String.t()]} | {:error, :closed | String.t()}
  def search(db_ref, query_vector, filter, k, ef_search \\ nil)

  def search(_db_ref, _query_vector, _filter, _k, _ef_search),
    do: :erlang.nif_error(:nif_not_loaded)

  @spec search_detailed(
          reference(),
          [number()],
          map() | keyword() | nil,
          non_neg_integer(),
          pos_integer() | nil
        ) ::
          {:ok, [map()]} | {:error, :closed | String.t()}
  def search_detailed(_db_ref, _query_vector, _filter, _k, _ef_search),
    do: :erlang.nif_error(:nif_not_loaded)

  @spec calculate_distance([number()], [number()], atom() | String.t()) ::
          {:ok, float()} | {:error, String.t()}
  def calculate_distance(_a, _b, _metric), do: :erlang.nif_error(:nif_not_loaded)

  @spec calculate_distance_dirty([number()], [number()], atom() | String.t()) ::
          {:ok, float()} | {:error, String.t()}
  def calculate_distance_dirty(_a, _b, _metric), do: :erlang.nif_error(:nif_not_loaded)

  @spec info(reference()) :: {:ok, map()} | {:error, :closed | String.t()}
  def info(_db_ref), do: :erlang.nif_error(:nif_not_loaded)
end
