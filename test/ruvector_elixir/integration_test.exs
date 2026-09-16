defmodule RuvectorElixir.IntegrationTest do
  use ExUnit.Case, async: true

  alias RuvectorElixir.VectorEntry

  describe "end-to-end workflows (matching upstream integration_tests.rs)" do
    test "complete insert and search workflow with HNSW" do
      db_path =
        Path.join(System.tmp_dir!(), "ruvector_hnsw_#{System.unique_integer([:positive])}.rvf")

      on_exit(fn -> File.rm_rf(db_path) end)

      options = %{
        metric: :cosine,
        hnsw: %{
          m: 16,
          ef_construction: 100,
          ef_search: 50,
          max_elements: 10_000
        }
      }

      {:ok, db} = RuvectorElixir.open(db_path, 128, options)

      # Insert 100 vectors
      entries =
        for i <- 0..99 do
          vec = for j <- 0..127, do: (i + j) * 0.01

          VectorEntry.new(vec,
            id: "vec_#{i}",
            metadata: %{"index" => i, "label" => "sample_#{i}"}
          )
        end

      assert {:ok, ids} = RuvectorElixir.insert_batch(db, entries)
      assert length(ids) == 100
      assert {:ok, 100} = RuvectorElixir.len(db)

      # Search for top 10
      query = for j <- 0..127, do: j * 0.01
      top_ids = RuvectorElixir.search(db, query, 10)
      assert length(top_ids) == 10
      assert "vec_0" in top_ids

      # Search detailed
      {:ok, detailed} = RuvectorElixir.search_detailed(db, query, nil, 10, 100)
      assert length(detailed) == 10
      first = hd(detailed)
      assert is_binary(first.id)
      assert is_float(first.score)
      assert is_list(first.vector)
      assert length(first.vector) == 128
      assert is_map(first.metadata)
      assert is_integer(first.metadata["index"])
    end

    test "persistence and reload across database handles" do
      db_path =
        Path.join(System.tmp_dir!(), "ruvector_persist_#{System.unique_integer([:positive])}.rvf")

      on_exit(fn -> File.rm_rf(db_path) end)

      # Create and populate
      {:ok, db1} = RuvectorElixir.open(db_path, 3)

      for i <- 0..9 do
        {:ok, _} =
          RuvectorElixir.insert(db1, %{
            id: "vec_#{i}",
            vector: [i * 1.0, i * 2.0, i * 3.0]
          })
      end

      assert {:ok, 10} = RuvectorElixir.len(db1)

      # Reload from disk using new handle
      {:ok, db2} = RuvectorElixir.open(db_path, 3)
      assert {:ok, 10} = RuvectorElixir.len(db2)

      {:ok, entry5} = RuvectorElixir.get(db2, "vec_5")
      assert entry5.vector == [5.0, 10.0, 15.0]

      query = [5.0, 10.0, 15.0]
      results = RuvectorElixir.search(db2, query, 3)
      assert hd(results) == "vec_5"
    end

    test "mixed operations: batch insert, deletions, single inserts, search" do
      db_path =
        Path.join(System.tmp_dir!(), "ruvector_mixed_#{System.unique_integer([:positive])}.rvf")

      on_exit(fn -> File.rm_rf(db_path) end)

      {:ok, db} = RuvectorElixir.open(db_path, 64)

      # Insert initial batch of 50
      initial =
        for i <- 0..49 do
          %{
            id: "vec_#{i}",
            vector: for(j <- 0..63, do: (i + j) * 0.1)
          }
        end

      {:ok, _} = RuvectorElixir.insert_batch(db, initial)
      assert {:ok, 50} = RuvectorElixir.len(db)

      # Delete 10
      for i <- 0..9 do
        assert {:ok, true} = RuvectorElixir.delete(db, "vec_#{i}")
      end

      assert {:ok, 40} = RuvectorElixir.len(db)

      # Insert 10 more
      for i <- 50..59 do
        {:ok, _} =
          RuvectorElixir.insert(db, %{
            id: "vec_#{i}",
            vector: for(j <- 0..63, do: (i + j) * 0.1)
          })
      end

      assert {:ok, 50} = RuvectorElixir.len(db)

      # Search
      query = for(j <- 0..63, do: j * 0.1)
      results = RuvectorElixir.search(db, query, 20)
      assert length(results) == 20
      # Deleted vectors 0..9 should not be in results
      for i <- 0..9 do
        refute "vec_#{i}" in results
      end
    end

    test "all distance metrics: Euclidean, Cosine, Dot Product, Manhattan" do
      metrics = [:euclidean, :cosine, :dot_product, :manhattan]

      for metric <- metrics do
        db_path =
          Path.join(
            System.tmp_dir!(),
            "ruvector_metric_#{metric}_#{System.unique_integer([:positive])}.rvf"
          )

        on_exit(fn -> File.rm_rf(db_path) end)

        {:ok, db} = RuvectorElixir.open(db_path, 32, %{metric: metric})

        # Insert 20 vectors
        for i <- 0..19 do
          vec = for j <- 0..31, do: (i + j) * 0.1
          {:ok, _} = RuvectorElixir.insert(db, %{id: "vec_#{i}", vector: vec})
        end

        # Search
        query = for j <- 0..31, do: j * 0.1
        results = RuvectorElixir.search(db, query, 5)
        assert length(results) == 5

        # Verify detailed search scores are sorted ascending
        {:ok, detailed} = RuvectorElixir.search_detailed(db, query, 5)
        assert length(detailed) == 5

        scores = Enum.map(detailed, & &1.score)
        assert scores == Enum.sort(scores), "Scores not sorted ascending for #{metric}"
      end
    end

    test "HNSW with different configurations (M=8, M=16, M=32)" do
      configs = [
        %{m: 8, ef_construction: 50, ef_search: 50, max_elements: 1000},
        %{m: 16, ef_construction: 100, ef_search: 100, max_elements: 1000},
        %{m: 32, ef_construction: 200, ef_search: 200, max_elements: 1000}
      ]

      for config <- configs do
        db_path =
          Path.join(
            System.tmp_dir!(),
            "ruvector_hnsw_m#{config.m}_#{System.unique_integer([:positive])}.rvf"
          )

        on_exit(fn -> File.rm_rf(db_path) end)

        {:ok, db} = RuvectorElixir.open(db_path, 64, %{hnsw: config})

        entries =
          for i <- 0..99 do
            %{
              id: "vec_#{i}",
              vector: for(j <- 0..63, do: (i + j) * 0.01)
            }
          end

        {:ok, _} = RuvectorElixir.insert_batch(db, entries)

        query = for j <- 0..63, do: j * 0.01
        results = RuvectorElixir.search(db, query, 10)
        assert length(results) == 10
      end
    end

    test "dimension validation on insert" do
      db_path =
        Path.join(
          System.tmp_dir!(),
          "ruvector_dim_val_#{System.unique_integer([:positive])}.rvf"
        )

      on_exit(fn -> File.rm_rf(db_path) end)

      {:ok, db} = RuvectorElixir.open(db_path, 64)

      # 3 dimensions when 64 expected
      assert {:error, reason} = RuvectorElixir.insert(db, [1.0, 2.0, 3.0])
      assert String.contains?(reason, "mismatch") or String.contains?(reason, "Dimension")
    end

    test "dimension validation on search query (matching upstream test_search_with_wrong_dimension)" do
      db_path =
        Path.join(
          System.tmp_dir!(),
          "ruvector_search_dim_#{System.unique_integer([:positive])}.rvf"
        )

      on_exit(fn -> File.rm_rf(db_path) end)

      {:ok, db} = RuvectorElixir.open(db_path, 64)
      {:ok, _} = RuvectorElixir.insert(db, %{id: "v1", vector: List.duplicate(0.5, 64)})

      # Query with only 3 dimensions when 64 expected
      query = [1.0, 2.0, 3.0]
      assert {:error, reason} = RuvectorElixir.search(db, query, 10)
      assert String.contains?(reason, "mismatch") or String.contains?(reason, "Dimension")

      assert {:error, reason_det} = RuvectorElixir.search_detailed(db, query, 10)
      assert String.contains?(reason_det, "mismatch") or String.contains?(reason_det, "Dimension")

      assert_raise ArgumentError, fn ->
        RuvectorElixir.search!(db, query, 10)
      end
    end

    test "open with keyword list options and open! helper" do
      db_path =
        Path.join(
          System.tmp_dir!(),
          "ruvector_kw_opts_#{System.unique_integer([:positive])}.rvf"
        )

      on_exit(fn -> File.rm_rf(db_path) end)

      # Pass keyword list with nested keyword list for HNSW
      db =
        RuvectorElixir.open!(db_path, 32,
          metric: :euclidean,
          hnsw: [m: 16, ef_construction: 50, ef_search: 50]
        )

      info = RuvectorElixir.info(db)
      assert info["dimensions"] == 32
      assert info["distance_metric"] == "euclidean"
      assert is_map(info["hnsw"])
      assert info["hnsw"]["m"] == 16

      # open! with invalid path raises
      assert_raise RuntimeError, ~r/Failed to open database/, fn ->
        RuvectorElixir.open!("/nonexistent_root_dir_abc_123/db.rvf", 32)
      end
    end

    test "open with unique path and storage_path in info" do
      db_path =
        Path.join(
          System.tmp_dir!(),
          "ruvector_info_path_#{System.unique_integer([:positive])}.rvf"
        )

      on_exit(fn -> File.rm_rf(db_path) end)

      {:ok, mem_db} = RuvectorElixir.open(db_path, 16, metric: :cosine)

      info = RuvectorElixir.info(mem_db)
      assert info["dimensions"] == 16
      assert info["storage_path"] == db_path

      {:ok, id} = RuvectorElixir.insert(mem_db, List.duplicate(0.1, 16))
      assert is_binary(id)
      assert {:ok, 1} = RuvectorElixir.len(mem_db)

      results = RuvectorElixir.search(mem_db, List.duplicate(0.1, 16), 1)
      assert results == [id]
    end

    test "open with quantization options (:scalar, :binary, :none)" do
      for q <- [:scalar, :binary, :none] do
        db_path =
          Path.join(
            System.tmp_dir!(),
            "ruvector_quant_#{q}_#{System.unique_integer([:positive])}.rvf"
          )

        on_exit(fn -> File.rm_rf(db_path) end)

        {:ok, db} = RuvectorElixir.open(db_path, 16, quantization: q)
        assert {:ok, _id} = RuvectorElixir.insert(db, List.duplicate(1.0, 16))
        assert {:ok, 1} = RuvectorElixir.len(db)
      end
    end

    test "batch operations with 1,000 vectors (matching upstream scale tests)" do
      db_path =
        Path.join(
          System.tmp_dir!(),
          "ruvector_1k_#{System.unique_integer([:positive])}.rvf"
        )

      on_exit(fn -> File.rm_rf(db_path) end)

      {:ok, db} = RuvectorElixir.open(db_path, 128, hnsw: true)

      entries =
        for i <- 0..999 do
          %{
            id: "vec_#{i}",
            vector: for(j <- 0..127, do: (i + j) * 0.001),
            metadata: %{"group" => rem(i, 10)}
          }
        end

      assert {:ok, ids} = RuvectorElixir.insert_batch(db, entries)
      assert length(ids) == 1000
      assert {:ok, 1000} = RuvectorElixir.len(db)

      query = for(j <- 0..127, do: j * 0.001)
      results = RuvectorElixir.search(db, query, 10)
      assert length(results) == 10
      assert hd(results) == "vec_0"
    end
  end
end
