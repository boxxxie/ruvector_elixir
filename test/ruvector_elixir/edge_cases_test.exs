defmodule RuvectorElixir.EdgeCasesTest do
  use ExUnit.Case, async: true

  setup do
    db_path =
      Path.join(System.tmp_dir!(), "ruvector_edge_#{System.unique_integer([:positive])}.rvf")

    on_exit(fn -> File.rm_rf(db_path) end)

    {:ok, db} = RuvectorElixir.open(db_path, 4)
    {:ok, db: db, db_path: db_path}
  end

  describe "boundary and edge case tests" do
    test "inserting empty vector returns dimension mismatch", %{db: db} do
      assert {:error, reason} = RuvectorElixir.insert(db, [])
      assert String.contains?(reason, "mismatch") or String.contains?(reason, "Dimension")
    end

    test "inserting vector with partial dimensions returns error", %{db: db} do
      assert {:error, reason} = RuvectorElixir.insert(db, [1.0, 2.0])
      assert String.contains?(reason, "mismatch") or String.contains?(reason, "Dimension")
    end

    test "inserting vector with excess dimensions returns error", %{db: db} do
      assert {:error, reason} = RuvectorElixir.insert(db, [1.0, 2.0, 3.0, 4.0, 5.0])
      assert String.contains?(reason, "mismatch") or String.contains?(reason, "Dimension")
    end

    test "upserting existing ID overwrites the entry", %{db: db} do
      assert {:ok, "dup_id"} =
               RuvectorElixir.insert(db, %{
                 id: "dup_id",
                 vector: [1.0, 1.0, 1.0, 1.0],
                 metadata: %{"version" => 1}
               })

      assert {:ok, 1} = RuvectorElixir.len(db)

      # Re-insert with new vector and metadata
      assert {:ok, "dup_id"} =
               RuvectorElixir.insert(db, %{
                 id: "dup_id",
                 vector: [2.0, 2.0, 2.0, 2.0],
                 metadata: %{"version" => 2}
               })

      # Count should still be 1
      assert {:ok, 1} = RuvectorElixir.len(db)

      {:ok, updated} = RuvectorElixir.get(db, "dup_id")
      assert updated.vector == [2.0, 2.0, 2.0, 2.0]
      assert updated.metadata["version"] == 2
    end

    test "search on completely empty database returns empty list", %{db: db} do
      results = RuvectorElixir.search(db, [0.0, 0.0, 0.0, 0.0], 10)
      assert results == []

      {:ok, detailed} = RuvectorElixir.search_detailed(db, [0.0, 0.0, 0.0, 0.0], 10)
      assert detailed == []
    end

    test "search with k = 0 returns empty list", %{db: db} do
      RuvectorElixir.insert(db, %{id: "e1", vector: [1.0, 0.0, 0.0, 0.0]})
      results = RuvectorElixir.search(db, [1.0, 0.0, 0.0, 0.0], 0)
      assert results == []
    end

    test "vectors with negative and float values", %{db: db} do
      neg_vec = [-1.5, -0.25, 0.0, 3.75]
      assert {:ok, "neg_id"} = RuvectorElixir.insert(db, %{id: "neg_id", vector: neg_vec})

      {:ok, retrieved} = RuvectorElixir.get(db, "neg_id")
      assert retrieved.vector == neg_vec

      {:ok, dist} = RuvectorElixir.calculate_distance(neg_vec, neg_vec, :euclidean)
      assert_in_delta dist, 0.0, 0.001
    end

    test "database info metadata on opened DB", %{db_path: db_path} do
      options = %{
        metric: :manhattan,
        hnsw: true
      }

      custom_path = db_path <> "_custom.rvf"
      on_exit(fn -> File.rm_rf(custom_path) end)

      {:ok, db} = RuvectorElixir.open(custom_path, 16, options)
      info = RuvectorElixir.info(db)

      assert info["dimensions"] == 16
      assert info["node_count"] == 0
      assert info["distance_metric"] == "manhattan"
    end

    test "delete then re-insert same ID", %{db: db} do
      RuvectorElixir.insert(db, %{id: "transient", vector: [0.1, 0.2, 0.3, 0.4]})
      assert {:ok, 1} = RuvectorElixir.len(db)

      assert {:ok, true} = RuvectorElixir.delete(db, "transient")
      assert {:ok, 0} = RuvectorElixir.len(db)

      # Second delete returns false
      assert {:ok, false} = RuvectorElixir.delete(db, "transient")

      # Re-insert
      assert {:ok, "transient"} =
               RuvectorElixir.insert(db, %{id: "transient", vector: [0.5, 0.25, 0.75, 1.0]})

      assert {:ok, 1} = RuvectorElixir.len(db)
      {:ok, entry} = RuvectorElixir.get(db, "transient")
      assert entry.vector == [0.5, 0.25, 0.75, 1.0]
    end

    test "null and empty metadata serialization", %{db: db} do
      RuvectorElixir.insert(db, %{id: "null_meta", vector: [0.0, 0.0, 0.0, 0.0], metadata: nil})
      {:ok, e1} = RuvectorElixir.get(db, "null_meta")
      assert e1.metadata == nil

      RuvectorElixir.insert(db, %{id: "empty_meta", vector: [1.0, 1.0, 1.0, 1.0], metadata: %{}})
      {:ok, e2} = RuvectorElixir.get(db, "empty_meta")
      assert e2.metadata == %{}
    end

    test "invalid id format returns error instead of silently generating random UUID", %{db: db} do
      assert {:error, "Invalid vector entry format"} =
               RuvectorElixir.insert(db, %{id: 3.14, vector: [1.0, 2.0, 3.0, 4.0]})

      assert {:error, "Invalid vector entry format"} =
               RuvectorElixir.insert(db, %{id: {:bad, :tuple}, vector: [1.0, 2.0, 3.0, 4.0]})
    end

    test "invalid vector format returns error", %{db: db} do
      assert {:error, "Invalid vector entry format"} =
               RuvectorElixir.insert(db, %{id: "bad_vec", vector: ["a", "b", "c", "d"]})

      assert {:error, "Invalid vector entry format"} =
               RuvectorElixir.insert(db, %{id: "bad_vec2", vector: "not_a_vector"})
    end

    test "invalid metadata format returns error", %{db: db} do
      assert {:error, "Invalid vector entry format"} =
               RuvectorElixir.insert(db, %{
                 id: "bad_meta",
                 vector: [1.0, 1.0, 1.0, 1.0],
                 metadata: "not_a_map"
               })
    end
  end
end
