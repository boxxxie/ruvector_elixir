defmodule RuvectorElixir do
  @moduledoc """
  A Rustler NIF binding to the ruvector-core Rust crate.
  This allows querying the `.rvf` GraphRAG metadata directly from the BEAM.
  """

  use Rustler, otp_app: :ruvector_elixir, crate: "ruvector_nif"

  @doc """
  Opens a ruvector DB at the specified path. Returns an opaque ResourceArc reference.
  """
  def open_db(_path, _dimensions), do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Searches the opened DB for the top `k` closest vectors to the given `query_vector`.
  Accepts an optional map of metadata key-value filters.
  Returns a list of matching schema node IDs.
  """
  def search(db_ref, query_vector, k), do: search(db_ref, query_vector, nil, k)
  def search(_db_ref, _query_vector, _filter, _k), do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Gets metadata info from the opened DB (e.g. node_count).
  """
  def info(_db_ref), do: :erlang.nif_error(:nif_not_loaded)
end
