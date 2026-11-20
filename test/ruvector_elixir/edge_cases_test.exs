defmodule RuvectorElixir.EdgeCasesTest do
  use ExUnit.Case, async: true

  alias RuvectorElixir.{Error, VectorEntry}

  @moduletag :tmp_dir

  setup %{tmp_dir: tmp_dir} do
    db_path = Path.join(tmp_dir, "ruvector_edge.rvf")
    {:ok, db} = RuvectorElixir.open(db_path, 4)
    {:ok, db: db, db_path: db_path, tmp_dir: tmp_dir}
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

      assert {:ok, "dup_id"} =
               RuvectorElixir.insert(db, %{
                 id: "dup_id",
                 vector: [2.0, 2.0, 2.0, 2.0],
                 metadata: %{"version" => 2}
               })

      assert {:ok, 1} = RuvectorElixir.len(db)

      assert {:ok, updated} = RuvectorElixir.get(db, "dup_id")
      assert updated.vector == [2.0, 2.0, 2.0, 2.0]
      assert updated.metadata["version"] == 2
    end

    test "search on completely empty database returns empty list", %{db: db} do
      assert {:ok, []} = RuvectorElixir.search(db, [0.0, 0.0, 0.0, 0.0], 10)
      assert {:ok, []} = RuvectorElixir.search_detailed(db, [0.0, 0.0, 0.0, 0.0], 10)
    end

    test "search with k = 0 returns empty list", %{db: db} do
      assert {:ok, "e1"} = RuvectorElixir.insert(db, %{id: "e1", vector: [1.0, 0.0, 0.0, 0.0]})
      assert {:ok, []} = RuvectorElixir.search(db, [1.0, 0.0, 0.0, 0.0], 0)
    end

    test "zero-vector cosine insert and search does not produce NaN or crash", %{db: db} do
      assert {:ok, "zero"} =
               RuvectorElixir.insert(db, %{id: "zero", vector: [0.0, 0.0, 0.0, 0.0]})

      case RuvectorElixir.search_detailed(db, [0.0, 0.0, 0.0, 0.0], 5) do
        {:ok, results} ->
          assert Enum.all?(results, &is_float(&1.score))

        {:error, reason} ->
          assert is_binary(reason)
      end

      case RuvectorElixir.distance([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], :cosine) do
        {:ok, dist} -> assert is_float(dist)
        {:error, reason} -> assert is_binary(reason)
      end
    end

    test "vectors with negative and float values", %{db: db} do
      neg_vec = [-1.5, -0.25, 0.0, 3.75]
      assert {:ok, "neg_id"} = RuvectorElixir.insert(db, %{id: "neg_id", vector: neg_vec})

      assert {:ok, retrieved} = RuvectorElixir.get(db, "neg_id")
      assert retrieved.vector == neg_vec

      assert {:ok, dist} = RuvectorElixir.distance(neg_vec, neg_vec, :euclidean)
      assert_in_delta dist, 0.0, 0.001
    end

    test "database info metadata on opened DB", %{tmp_dir: tmp_dir} do
      custom_path = Path.join(tmp_dir, "ruvector_custom.rvf")

      assert {:ok, db} =
               RuvectorElixir.open(custom_path, 16, %{metric: :manhattan, hnsw: true})

      assert {:ok, info} = RuvectorElixir.info(db)
      assert info.dimensions == 16
      assert info.node_count == 0
      assert info.distance_metric == :manhattan
      assert info.storage_path == custom_path
      assert is_map(info.hnsw)
      assert info.quantization == nil
    end

    test "delete then re-insert same ID", %{db: db} do
      assert {:ok, "transient"} =
               RuvectorElixir.insert(db, %{id: "transient", vector: [0.1, 0.2, 0.3, 0.4]})

      assert {:ok, 1} = RuvectorElixir.len(db)

      assert {:ok, true} = RuvectorElixir.delete(db, "transient")
      assert {:ok, 0} = RuvectorElixir.len(db)

      assert {:ok, false} = RuvectorElixir.delete(db, "transient")

      assert {:ok, "transient"} =
               RuvectorElixir.insert(db, %{id: "transient", vector: [0.5, 0.25, 0.75, 1.0]})

      assert {:ok, 1} = RuvectorElixir.len(db)
      assert {:ok, entry} = RuvectorElixir.get(db, "transient")
      assert entry.vector == [0.5, 0.25, 0.75, 1.0]
    end

    test "null and empty metadata serialization", %{db: db} do
      assert {:ok, "null_meta"} =
               RuvectorElixir.insert(db, %{
                 id: "null_meta",
                 vector: [0.0, 0.0, 0.0, 0.0],
                 metadata: nil
               })

      assert {:ok, e1} = RuvectorElixir.get(db, "null_meta")
      assert e1.metadata == nil

      assert {:ok, "empty_meta"} =
               RuvectorElixir.insert(db, %{
                 id: "empty_meta",
                 vector: [1.0, 1.0, 1.0, 1.0],
                 metadata: %{}
               })

      assert {:ok, e2} = RuvectorElixir.get(db, "empty_meta")
      assert e2.metadata == %{}
    end

    test "rejects invalid, nil, and boolean IDs on insert, get, and delete", %{db: db} do
      assert {:error, "Invalid vector entry format"} =
               RuvectorElixir.insert(db, %{id: 3.14, vector: [1.0, 2.0, 3.0, 4.0]})

      assert {:error, "Invalid vector entry format"} =
               RuvectorElixir.insert(db, %{id: {:bad, :tuple}, vector: [1.0, 2.0, 3.0, 4.0]})

      assert {:error, "Invalid vector entry format"} =
               RuvectorElixir.insert(db, %{id: true, vector: [1.0, 2.0, 3.0, 4.0]})

      assert {:error, "Invalid vector entry format"} =
               RuvectorElixir.insert(db, %{id: false, vector: [1.0, 2.0, 3.0, 4.0]})

      # Insert a record literally named "nil" or "true" to ensure get/delete(nil/true) don't match it
      assert {:ok, "nil"} =
               RuvectorElixir.insert(db, %{id: "nil", vector: [1.0, 0.0, 0.0, 0.0]})

      assert {:error, "Invalid id format"} = RuvectorElixir.get(db, nil)
      assert {:error, "Invalid id format"} = RuvectorElixir.delete(db, nil)
      assert {:error, "Invalid id format"} = RuvectorElixir.get(db, true)
      assert {:error, "Invalid id format"} = RuvectorElixir.delete(db, false)
      assert {:error, "Invalid id format"} = RuvectorElixir.get(db, "")
      assert {:error, "Invalid id format"} = RuvectorElixir.delete(db, "")
      assert {:ok, _} = RuvectorElixir.get(db, "nil")

      assert {:error, "Invalid vector entry format"} =
               RuvectorElixir.insert(db, %{id: "", vector: [1.0, 0.0, 0.0, 0.0]})

      assert {:error, "Invalid vector entry format"} =
               RuvectorElixir.insert(db, %{
                 :id => "real",
                 "id" => nil,
                 :vector => [1.0, 0.0, 0.0, 0.0]
               })
    end

    test "rejects non-finite floats (f32 overflow) on insert and search", %{db: db} do
      assert {:error, "Invalid vector entry format"} =
               RuvectorElixir.insert(db, %{id: "inf", vector: [1.0e100, 0.0, 0.0, 0.0]})

      assert {:error, "Invalid query vector"} =
               RuvectorElixir.search(db, [1.0e100, 0.0, 0.0, 0.0], 5)

      assert {:error, "Invalid query vector"} =
               RuvectorElixir.search_detailed(db, [1.0e100, 0.0, 0.0, 0.0], 5)
    end

    test "rejects invalid vector format", %{db: db} do
      assert {:error, "Invalid vector entry format"} =
               RuvectorElixir.insert(db, %{id: "bad_vec", vector: ["a", "b", "c", "d"]})

      assert {:error, "Invalid vector entry format"} =
               RuvectorElixir.insert(db, %{id: "bad_vec2", vector: "not_a_vector"})
    end

    test "rejects invalid and unsupported metadata terms instead of coercing to nil", %{db: db} do
      assert {:error, "Invalid vector entry format"} =
               RuvectorElixir.insert(db, %{
                 id: "bad_meta",
                 vector: [1.0, 1.0, 1.0, 1.0],
                 metadata: "not_a_map"
               })

      assert {:error, "Invalid vector entry format"} =
               RuvectorElixir.insert(db, %{
                 id: "struct_meta",
                 vector: [1.0, 1.0, 1.0, 1.0],
                 metadata: ~D[2026-01-01]
               })

      assert {:error, "Invalid vector entry format"} =
               RuvectorElixir.insert(db, %{
                 id: "tuple_meta",
                 vector: [1.0, 1.0, 1.0, 1.0],
                 metadata: %{"bad" => {:a, :b}}
               })

      assert {:error, "Invalid vector entry format"} =
               RuvectorElixir.insert(db, %{
                 id: "pid_meta",
                 vector: [1.0, 1.0, 1.0, 1.0],
                 metadata: %{"bad" => self()}
               })

      assert {:error, "Invalid vector entry format"} =
               RuvectorElixir.insert(db, %{
                 id: "bad_bin_meta",
                 vector: [1.0, 1.0, 1.0, 1.0],
                 metadata: %{"bad" => <<0xFF, 0xFE>>}
               })

      assert {:error, "Invalid vector entry format"} =
               RuvectorElixir.insert(db, %{
                 id: "bad_key_meta",
                 vector: [1.0, 1.0, 1.0, 1.0],
                 metadata: %{{:tuple_key} => "v"}
               })

      assert {:error, "Invalid vector entry format"} =
               RuvectorElixir.insert(db, %{
                 id: "kw_meta",
                 vector: [1.0, 1.0, 1.0, 1.0],
                 metadata: %{"tags" => [env: "prod", tier: 1]}
               })

      deeply_nested =
        Enum.reduce(1..150, "leaf", fn _, acc -> %{"n" => acc} end)

      assert {:error, "Invalid vector entry format"} =
               RuvectorElixir.insert(db, %{
                 id: "deep_meta",
                 vector: [1.0, 1.0, 1.0, 1.0],
                 metadata: deeply_nested
               })

      assert {:error, "Vector batch must be a list"} =
               RuvectorElixir.Native.insert_batch(db, :not_a_list)
    end

    test "rejects malformed search filters instead of running unfiltered queries", %{db: db} do
      assert {:ok, _} =
               RuvectorElixir.insert(db, %{
                 id: "tenant_a",
                 vector: [1.0, 0.0, 0.0, 0.0],
                 metadata: %{"tenant" => "a"}
               })

      assert_raise ArgumentError, fn ->
        RuvectorElixir.search(db, [1.0, 0.0, 0.0, 0.0], 5, filter: "not_a_filter")
      end

      assert_raise ArgumentError, fn ->
        RuvectorElixir.search(db, [1.0, 0.0, 0.0, 0.0], 5, filter: 123)
      end

      assert_raise ArgumentError, fn ->
        RuvectorElixir.search_detailed(db, [1.0, 0.0, 0.0, 0.0], 5, filter: [1, 2, 3])
      end

      assert_raise ArgumentError, fn ->
        RuvectorElixir.search(db, [1.0, 0.0, 0.0, 0.0], 5, filter: ~D[2026-01-01])
      end

      # Unsupported values inside an otherwise valid filter map return an error from NIF
      assert {:error, "Invalid filter format"} =
               RuvectorElixir.search(db, [1.0, 0.0, 0.0, 0.0], 5, filter: %{"bad" => {:tuple}})
    end

    test "rejects invalid open/3 options strictly", %{tmp_dir: tmp_dir} do
      db_path = Path.join(tmp_dir, "invalid_opts.rvf")

      assert_raise ArgumentError, fn ->
        RuvectorElixir.open(db_path, 0)
      end

      assert_raise ArgumentError, fn ->
        RuvectorElixir.open("", 4)
      end

      assert_raise ArgumentError, fn ->
        RuvectorElixir.open(<<0xFF, 0xFE>>, 4)
      end

      assert_raise ArgumentError, fn ->
        RuvectorElixir.open(db_path, 4, "not_a_map")
      end

      assert_raise ArgumentError, fn ->
        RuvectorElixir.open(db_path, 4, ~D[2026-01-01])
      end

      assert_raise ArgumentError, fn ->
        RuvectorElixir.open(db_path, 4, metric: :l2)
      end

      assert_raise ArgumentError, fn ->
        RuvectorElixir.open(db_path, 4, metric: :cosine, distance_metric: :euclidean)
      end

      assert_raise ArgumentError, fn ->
        RuvectorElixir.open(db_path, 4, unknown_opt: true)
      end

      assert_raise ArgumentError, fn ->
        RuvectorElixir.open(db_path, 4, hnsw: [m: -1])
      end

      assert_raise ArgumentError, fn ->
        RuvectorElixir.open(db_path, 4, hnsw: [m: 1])
      end

      assert_raise ArgumentError, fn ->
        RuvectorElixir.open(db_path, 4, hnsw: [m: 257])
      end

      assert_raise ArgumentError, fn ->
        RuvectorElixir.open(db_path, 4, quantization: :invalid)
      end

      assert_raise ArgumentError, fn ->
        RuvectorElixir.open(db_path, 4, quantization: %{type: :none, subspaces: 8})
      end

      assert_raise ArgumentError, fn ->
        RuvectorElixir.open(db_path, 4, quantization: %{type: :product, subspaces: 3, k: 256})
      end
    end

    test "close/1 releases handle and subsequent operations return {:error, :closed}", %{
      tmp_dir: tmp_dir
    } do
      db_path = Path.join(tmp_dir, "closable.rvf")
      assert {:ok, handle} = RuvectorElixir.open(db_path, 4)

      assert {:ok, "k1"} =
               RuvectorElixir.insert(handle, %{id: "k1", vector: [1.0, 0.0, 0.0, 0.0]})

      assert :ok = RuvectorElixir.close(handle)
      # Idempotent close
      assert :ok = RuvectorElixir.close(handle)
      assert :ok = RuvectorElixir.close!(handle)

      assert {:error, :closed} = RuvectorElixir.insert(handle, [1.0, 0.0, 0.0, 0.0])
      assert {:error, :closed} = RuvectorElixir.get(handle, "k1")
      assert {:error, :closed} = RuvectorElixir.delete(handle, "k1")
      assert {:error, :closed} = RuvectorElixir.len(handle)
      assert {:error, :closed} = RuvectorElixir.keys(handle)
      assert {:error, :closed} = RuvectorElixir.search(handle, [1.0, 0.0, 0.0, 0.0], 1)
      assert {:error, :closed} = RuvectorElixir.search_detailed(handle, [1.0, 0.0, 0.0, 0.0], 1)
      assert {:error, :closed} = RuvectorElixir.info(handle)

      assert_raise Error, ~r/closed/i, fn ->
        RuvectorElixir.empty?(handle)
      end

      assert_raise Error, ~r/closed/i, fn ->
        RuvectorElixir.info!(handle)
      end
    end

    test "memory:// database works without creating files on disk", %{tmp_dir: tmp_dir} do
      prev_cwd = File.cwd!()

      try do
        File.cd!(tmp_dir)
        assert {:ok, mem_db} = RuvectorElixir.open("memory://transient", 4, metric: :euclidean)

        assert {:ok, "m1"} =
                 RuvectorElixir.insert(mem_db, %{
                   id: "m1",
                   vector: [1.0, 0.0, 0.0, 0.0],
                   metadata: %{"kind" => "ephemeral"}
                 })

        assert {:ok, 1} = RuvectorElixir.len(mem_db)
        assert {:ok, ["m1"]} = RuvectorElixir.search(mem_db, [1.0, 0.0, 0.0, 0.0], 1)

        assert {:ok, [res]} =
                 RuvectorElixir.search_detailed(mem_db, [1.0, 0.0, 0.0, 0.0], 1,
                   filter: %{"kind" => "ephemeral"}
                 )

        assert res.id == "m1"
        assert {:ok, info} = RuvectorElixir.info(mem_db)
        assert info.storage_path == "memory://transient"

        refute File.exists?(Path.join(tmp_dir, "memory:"))
        assert :ok = RuvectorElixir.close(mem_db)
      after
        File.cd!(prev_cwd)
      end
    end

    test "VectorEntry.new/2 and VectorEntry.from_map/1 enforce ID and metadata invariants" do
      assert_raise ArgumentError, fn ->
        VectorEntry.new([1.0, 2.0], id: 3.14)
      end

      assert_raise ArgumentError, fn ->
        VectorEntry.new([1.0, 2.0], id: true)
      end

      assert_raise ArgumentError, fn ->
        VectorEntry.new([1.0, 2.0], metadata: ~D[2026-01-01])
      end

      entry = VectorEntry.from_map(%{id: 42, vector: [1.0, 2.0], metadata: nil})
      assert entry.id == "42"
      assert entry.vector == [1.0, 2.0]
    end
  end
end
