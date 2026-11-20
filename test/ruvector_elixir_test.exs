defmodule RuvectorElixirTest do
  use ExUnit.Case, async: true

  doctest RuvectorElixir.VectorEntry

  @tag :tmp_dir
  test "can open a database and search", %{tmp_dir: tmp_dir} do
    db_path = Path.join(tmp_dir, "ruvector_test.rvf")
    assert {:ok, db_ref} = RuvectorElixir.open(db_path, 768)
    assert is_reference(db_ref)

    query_vector = List.duplicate(0.1, 768)
    assert {:ok, []} = RuvectorElixir.search(db_ref, query_vector, 5)
  end

  @tag :tmp_dir
  test "search/4 accepts filter and works on empty DB", %{tmp_dir: tmp_dir} do
    db_path = Path.join(tmp_dir, "ruvector_filter_empty.rvf")
    assert {:ok, db_ref} = RuvectorElixir.open(db_path, 768)
    query_vector = List.duplicate(0.1, 768)

    assert {:ok, []} =
             RuvectorElixir.search(db_ref, query_vector, 5,
               filter: %{"schemaCleanName" => "diligence_analysis"}
             )
  end

  @tag :tmp_dir
  test "info/1 returns database metadata with atom keys", %{tmp_dir: tmp_dir} do
    db_path = Path.join(tmp_dir, "ruvector_info.rvf")
    assert {:ok, db_ref} = RuvectorElixir.open(db_path, 768)
    assert {:ok, info} = RuvectorElixir.info(db_ref)

    assert is_map(info)
    assert info.dimensions == 768
    assert info.node_count == 0
    assert info.distance_metric == :cosine
    assert info.storage_path == db_path
    assert info.hnsw == nil
    assert info.quantization == nil
  end
end
