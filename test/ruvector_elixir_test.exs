defmodule RuvectorElixirTest do
  use ExUnit.Case, async: true

  setup do
    db_path =
      Path.join(System.tmp_dir!(), "ruvector_test_#{System.unique_integer([:positive])}.db")

    on_exit(fn ->
      File.rm_rf!(db_path)
    end)

    {:ok, db_path: db_path}
  end

  test "can open a database and fail/succeed accordingly", %{db_path: db_path} do
    assert {:ok, db_ref} = RuvectorElixir.open_db(db_path, 768)

    assert is_reference(db_ref) or is_struct(db_ref) or
             (is_map(db_ref) and Map.has_key?(db_ref, :__struct__))

    # We should be able to search on the open database
    # A vector of 768 dimensions (since we hardcoded 768 for gemini-embedding-2)
    query_vector = List.duplicate(0.1, 768)
    results = RuvectorElixir.search(db_ref, query_vector, 5)
    assert is_list(results)
  end

  test "search/4 accepts filter and works on empty DB", %{db_path: db_path} do
    assert {:ok, db_ref} = RuvectorElixir.open_db(db_path, 768)
    query_vector = List.duplicate(0.1, 768)

    results =
      RuvectorElixir.search(db_ref, query_vector, %{"schemaCleanName" => "diligence_analysis"}, 5)

    assert results == []
  end

  test "info/1 returns database metadata", %{db_path: db_path} do
    assert {:ok, db_ref} = RuvectorElixir.open_db(db_path, 768)
    info = RuvectorElixir.info(db_ref)

    assert is_map(info)
    assert info["dimensions"] == 768
    assert info["node_count"] == 0
    assert info["distance_metric"] == "cosine"
  end
end
