defmodule RuvectorElixir.IntegrationTest do
  use ExUnit.Case, async: true

  alias RuvectorElixir.{Error, VectorEntry}

  @moduletag :tmp_dir

  describe "end-to-end workflows (matching upstream integration_tests.rs)" do
    test "complete insert and search workflow with HNSW", %{tmp_dir: tmp_dir} do
      db_path = Path.join(tmp_dir, "ruvector_hnsw.rvf")

      options = %{
        metric: :cosine,
        hnsw: %{
          m: 16,
          ef_construction: 100,
          ef_search: 50,
          max_elements: 10_000
        }
      }

      assert {:ok, db} = RuvectorElixir.open(db_path, 128, options)
      on_exit(fn -> RuvectorElixir.close(db) end)

      entries =
        for i <- 0..99 do
          vec = for j <- 0..127, do: :math.cos((i + 1) * (j + 1) * 0.13)

          VectorEntry.new(vec,
            id: "vec_#{i}",
            metadata: %{"index" => i, "label" => "sample_#{i}"}
          )
        end

      assert {:ok, ids} = RuvectorElixir.insert_batch(db, entries)
      assert length(ids) == 100
      assert {:ok, 100} = RuvectorElixir.len(db)

      query = for j <- 0..127, do: :math.cos((j + 1) * 0.13)
      assert {:ok, top_ids} = RuvectorElixir.search(db, query, 10)
      assert length(top_ids) == 10
      assert "vec_0" in top_ids

      assert {:ok, detailed} = RuvectorElixir.search_detailed(db, query, 10, ef_search: 100)
      assert length(detailed) == 10
      first = hd(detailed)
      assert is_binary(first.id)
      assert is_float(first.score)
      assert is_list(first.vector)
      assert length(first.vector) == 128
      assert is_map(first.metadata)
      assert is_integer(first.metadata["index"])
    end

    test "persistence and reload across database handles", %{tmp_dir: tmp_dir} do
      db_path = Path.join(tmp_dir, "ruvector_persist.rvf")

      assert {:ok, db1} = RuvectorElixir.open(db_path, 3)

      for i <- 0..9 do
        assert {:ok, _} =
                 RuvectorElixir.insert(db1, %{
                   id: "vec_#{i}",
                   vector: [i * 1.0, i * 2.0, i * 3.0]
                 })
      end

      assert {:ok, 10} = RuvectorElixir.len(db1)
      assert :ok = RuvectorElixir.close(db1)

      assert {:ok, db2} = RuvectorElixir.open(db_path, 3)
      on_exit(fn -> RuvectorElixir.close(db2) end)
      assert {:ok, 10} = RuvectorElixir.len(db2)

      assert {:ok, entry5} = RuvectorElixir.get(db2, "vec_5")
      assert entry5.vector == [5.0, 10.0, 15.0]

      query = [5.0, 10.0, 15.0]
      assert {:ok, results} = RuvectorElixir.search(db2, query, 3)
      assert hd(results) == "vec_5"
    end

    test "reopening .rvf database preserves persisted metric, HNSW, quantization, and dimensions even when omitted or mismatched",
         %{tmp_dir: tmp_dir} do
      db_path = Path.join(tmp_dir, "ruvector_reopen_opts.rvf")

      assert {:ok, db1} =
               RuvectorElixir.open(db_path, 8,
                 metric: :euclidean,
                 hnsw: [m: 8, ef_search: 32],
                 quantization: :scalar
               )

      assert {:ok, "v1"} =
               RuvectorElixir.insert(db1, %{
                 id: "v1",
                 vector: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
               })

      assert :ok = RuvectorElixir.close(db1)

      # Reopen with omitted options
      assert {:ok, db2} = RuvectorElixir.open(db_path, 8)
      assert {:ok, info2} = RuvectorElixir.info(db2)
      assert info2.dimensions == 8
      assert info2.distance_metric == :euclidean
      assert info2.hnsw.m == 8
      assert info2.hnsw.ef_search == 32
      assert info2.quantization == :scalar

      assert {:ok, ["v1"]} =
               RuvectorElixir.search(db2, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 1)

      assert :ok = RuvectorElixir.close(db2)

      # Reopen with mismatched dimensions -> persisted 8D config still wins
      assert {:ok, db3} = RuvectorElixir.open(db_path, 99)
      on_exit(fn -> RuvectorElixir.close(db3) end)
      assert {:ok, info3} = RuvectorElixir.info(db3)
      assert info3.dimensions == 8
      assert info3.distance_metric == :euclidean
      assert info3.quantization == :scalar
      assert {:ok, entry} = RuvectorElixir.get(db3, "v1")
      assert entry.id == "v1"

      assert {:ok, ["v1"]} =
               RuvectorElixir.search(db3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 1)
    end

    test "mixed operations: batch insert, deletions, single inserts, search", %{tmp_dir: tmp_dir} do
      db_path = Path.join(tmp_dir, "ruvector_mixed.rvf")
      assert {:ok, db} = RuvectorElixir.open(db_path, 64)
      on_exit(fn -> RuvectorElixir.close(db) end)

      initial =
        for i <- 0..49 do
          %{
            id: "vec_#{i}",
            vector: for(j <- 0..63, do: (i + j) * 0.1)
          }
        end

      assert {:ok, _} = RuvectorElixir.insert_batch(db, initial)
      assert {:ok, 50} = RuvectorElixir.len(db)

      for i <- 0..9 do
        assert {:ok, true} = RuvectorElixir.delete(db, "vec_#{i}")
      end

      assert {:ok, 40} = RuvectorElixir.len(db)

      for i <- 50..59 do
        assert {:ok, _} =
                 RuvectorElixir.insert(db, %{
                   id: "vec_#{i}",
                   vector: for(j <- 0..63, do: (i + j) * 0.1)
                 })
      end

      assert {:ok, 50} = RuvectorElixir.len(db)

      query = for(j <- 0..63, do: j * 0.1)
      assert {:ok, results} = RuvectorElixir.search(db, query, 20)
      assert length(results) == 20

      for i <- 0..9 do
        refute "vec_#{i}" in results
      end
    end

    test "all distance metrics: Euclidean, Cosine, Dot Product, Manhattan", %{tmp_dir: tmp_dir} do
      metrics = [:euclidean, :cosine, :dot_product, :manhattan]

      for metric <- metrics do
        db_path = Path.join(tmp_dir, "ruvector_metric_#{metric}.rvf")
        assert {:ok, db} = RuvectorElixir.open(db_path, 32, %{metric: metric})

        for i <- 0..19 do
          vec = for j <- 0..31, do: (i + j) * 0.1
          assert {:ok, _} = RuvectorElixir.insert(db, %{id: "vec_#{i}", vector: vec})
        end

        query = for j <- 0..31, do: j * 0.1
        assert {:ok, results} = RuvectorElixir.search(db, query, 5)
        assert length(results) == 5

        assert {:ok, detailed} = RuvectorElixir.search_detailed(db, query, 5)
        assert length(detailed) == 5

        scores = Enum.map(detailed, & &1.score)
        assert scores == Enum.sort(scores), "Scores not sorted ascending for #{metric}"
        assert :ok = RuvectorElixir.close(db)
      end
    end

    test "HNSW with different configurations (M=8, M=16, M=32)", %{tmp_dir: tmp_dir} do
      configs = [
        %{m: 8, ef_construction: 50, ef_search: 50, max_elements: 1000},
        %{m: 16, ef_construction: 100, ef_search: 100, max_elements: 1000},
        %{m: 32, ef_construction: 200, ef_search: 200, max_elements: 1000}
      ]

      for config <- configs do
        db_path = Path.join(tmp_dir, "ruvector_hnsw_m#{config.m}.rvf")
        assert {:ok, db} = RuvectorElixir.open(db_path, 64, %{hnsw: config})

        entries =
          for i <- 0..99 do
            %{
              id: "vec_#{i}",
              vector: for(j <- 0..63, do: (i + j) * 0.01)
            }
          end

        assert {:ok, _} = RuvectorElixir.insert_batch(db, entries)

        query = for j <- 0..63, do: j * 0.01
        assert {:ok, results} = RuvectorElixir.search(db, query, 10)
        assert length(results) == 10
        assert :ok = RuvectorElixir.close(db)
      end
    end

    test "dimension validation on insert", %{tmp_dir: tmp_dir} do
      db_path = Path.join(tmp_dir, "ruvector_dim_val.rvf")
      assert {:ok, db} = RuvectorElixir.open(db_path, 64)
      on_exit(fn -> RuvectorElixir.close(db) end)

      assert {:error, reason} = RuvectorElixir.insert(db, [1.0, 2.0, 3.0])
      assert String.contains?(reason, "mismatch") or String.contains?(reason, "Dimension")
    end

    test "dimension validation on search query (matching upstream test_search_with_wrong_dimension)",
         %{tmp_dir: tmp_dir} do
      db_path = Path.join(tmp_dir, "ruvector_search_dim.rvf")
      assert {:ok, db} = RuvectorElixir.open(db_path, 64)
      on_exit(fn -> RuvectorElixir.close(db) end)
      assert {:ok, _} = RuvectorElixir.insert(db, %{id: "v1", vector: List.duplicate(0.5, 64)})

      query = [1.0, 2.0, 3.0]
      assert {:error, reason} = RuvectorElixir.search(db, query, 10)
      assert String.contains?(reason, "mismatch") or String.contains?(reason, "Dimension")

      assert {:error, reason_det} = RuvectorElixir.search_detailed(db, query, 10)
      assert String.contains?(reason_det, "mismatch") or String.contains?(reason_det, "Dimension")

      assert_raise Error, fn ->
        RuvectorElixir.search!(db, query, 10)
      end
    end

    test "open with keyword list options and open! helper", %{tmp_dir: tmp_dir} do
      db_path = Path.join(tmp_dir, "ruvector_kw_opts.rvf")

      db =
        RuvectorElixir.open!(db_path, 32,
          metric: :euclidean,
          hnsw: [m: 16, ef_construction: 50, ef_search: 50]
        )

      on_exit(fn -> RuvectorElixir.close(db) end)

      assert {:ok, info} = RuvectorElixir.info(db)
      assert info.dimensions == 32
      assert info.distance_metric == :euclidean
      assert is_map(info.hnsw)
      assert info.hnsw.m == 16

      assert_raise Error, fn ->
        RuvectorElixir.open!("/nonexistent_root_dir_abc_123/db.rvf", 32)
      end
    end

    test "memory:// URI opens genuine in-memory database", %{} do
      assert {:ok, mem_db} = RuvectorElixir.open("memory://integration", 16, metric: :cosine)
      on_exit(fn -> RuvectorElixir.close(mem_db) end)

      assert {:ok, info} = RuvectorElixir.info(mem_db)
      assert info.dimensions == 16
      assert info.storage_path == "memory://integration"

      assert {:ok, id} = RuvectorElixir.insert(mem_db, List.duplicate(0.1, 16))
      assert is_binary(id)
      assert {:ok, 1} = RuvectorElixir.len(mem_db)

      assert {:ok, [^id]} = RuvectorElixir.search(mem_db, List.duplicate(0.1, 16), 1)
    end

    test "open with quantization options (:scalar, :binary, :none, and :product)", %{
      tmp_dir: tmp_dir
    } do
      for q <- [:scalar, :binary, :none] do
        db_path = Path.join(tmp_dir, "ruvector_quant_#{q}.rvf")

        assert {:ok, db} = RuvectorElixir.open(db_path, 16, quantization: q)
        assert {:ok, _id} = RuvectorElixir.insert(db, List.duplicate(1.0, 16))
        assert {:ok, 1} = RuvectorElixir.len(db)
        assert {:ok, info} = RuvectorElixir.info(db)
        assert info.quantization == q
        assert :ok = RuvectorElixir.close(db)
      end

      pq_path = Path.join(tmp_dir, "ruvector_quant_product.rvf")

      assert {:ok, pq_db} =
               RuvectorElixir.open(pq_path, 16,
                 quantization: %{type: :product, subspaces: 8, k: 256}
               )

      on_exit(fn -> RuvectorElixir.close(pq_db) end)

      assert {:ok, _id} = RuvectorElixir.insert(pq_db, List.duplicate(1.0, 16))
      assert {:ok, info} = RuvectorElixir.info(pq_db)
      assert info.quantization == %{type: :product, subspaces: 8, k: 256}
    end

    test "batch operations with 1,000 vectors (matching upstream scale tests)", %{
      tmp_dir: tmp_dir
    } do
      db_path = Path.join(tmp_dir, "ruvector_1k.rvf")
      assert {:ok, db} = RuvectorElixir.open(db_path, 128, hnsw: true)
      on_exit(fn -> RuvectorElixir.close(db) end)

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
      assert {:ok, results} = RuvectorElixir.search(db, query, 10)
      assert length(results) == 10
      assert hd(results) == "vec_0"
    end
  end
end
