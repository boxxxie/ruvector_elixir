defmodule RuvectorElixir.EdgeCasesTest do
  use ExUnit.Case, async: true

  alias RuvectorElixir.{Error, VectorEntry}

  @moduletag :tmp_dir

  setup %{tmp_dir: tmp_dir} do
    db_path = Path.join(tmp_dir, "ruvector_edge.rvf")
    {:ok, db} = RuvectorElixir.open(db_path, 4)
    on_exit(fn -> RuvectorElixir.close(db) end)
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

      on_exit(fn -> RuvectorElixir.close(db) end)

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
               RuvectorElixir.insert_batch(db, :not_a_list)

      assert {:error, "Vector batch must be a list"} =
               RuvectorElixir.insert_batch(db, [[1.0, 2.0, 3.0, 4.0] | :bad])
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

      for invalid_filter <- [
            [true: "x"],
            [false: "x"],
            [nil: "x"],
            [__struct__: Date],
            [{<<0xFF, 0xFE>>, "x"}],
            [{:a, 1} | :bad],
            [{:a, 1} | {:b, 2}],
            [{1, "a"}],
            %{1 => "a"},
            %{__struct__: "fake"}
          ] do
        assert_raise ArgumentError, fn ->
          RuvectorElixir.search(db, [1.0, 0.0, 0.0, 0.0], 5, filter: invalid_filter)
        end

        assert_raise ArgumentError, fn ->
          RuvectorElixir.search_detailed(db, [1.0, 0.0, 0.0, 0.0], 5, filter: invalid_filter)
        end
      end

      assert {:error, "Invalid filter format"} =
               RuvectorElixir.Native.search(db, [1.0, 0.0, 0.0, 0.0], %{1 => "a"}, 5, nil)

      assert {:error, "Invalid filter format"} =
               RuvectorElixir.Native.search(db, [1.0, 0.0, 0.0, 0.0], [{1, "a"}], 5, nil)

      assert {:error, "Invalid vector entry format"} =
               RuvectorElixir.insert(db, %{
                 id: "int_key_meta",
                 vector: [1.0, 0.0, 0.0, 0.0],
                 metadata: %{1 => "a"}
               })

      assert {:error, "Invalid filter format"} =
               RuvectorElixir.search(db, [1.0, 0.0, 0.0, 0.0], 5,
                 filter: [{"tenant", "a"}, {:tenant, "b"}]
               )

      assert {:error, "Invalid filter format"} =
               RuvectorElixir.search(db, [1.0, 0.0, 0.0, 0.0], 5,
                 filter: %{"tenant" => "a", :tenant => "b"}
               )

      # Unsupported values inside an otherwise valid filter map return an error from NIF
      assert {:error, "Invalid filter format"} =
               RuvectorElixir.search(db, [1.0, 0.0, 0.0, 0.0], 5, filter: %{"bad" => {:tuple}})
    end

    test "enforces symmetric 127-level JSON container nesting across metadata, map filters, and list filters",
         %{db: db} do
      depth_127_map = Enum.reduce(1..127, "leaf", fn _, acc -> %{"n" => acc} end)
      depth_128_map = Enum.reduce(1..128, "leaf", fn _, acc -> %{"n" => acc} end)
      depth_128_list_leaf = Enum.reduce(1..127, [], fn _, acc -> %{"n" => acc} end)
      depth_127_kw = [n: Enum.reduce(1..126, "leaf", fn _, acc -> %{"n" => acc} end)]
      depth_128_kw = [n: Enum.reduce(1..127, "leaf", fn _, acc -> %{"n" => acc} end)]

      entry_127 =
        VectorEntry.new([1.0, 0.0, 0.0, 0.0], id: "deep_127", metadata: depth_127_map)

      assert {:ok, "deep_127"} = RuvectorElixir.insert(db, entry_127)
      assert {:ok, fetched} = RuvectorElixir.get(db, "deep_127")
      assert fetched.metadata == depth_127_map

      assert {:ok, ["deep_127"]} =
               RuvectorElixir.search(db, [1.0, 0.0, 0.0, 0.0], 5, filter: depth_127_map)

      assert {:ok, ["deep_127"]} =
               RuvectorElixir.search(db, [1.0, 0.0, 0.0, 0.0], 5, filter: depth_127_kw)

      for too_deep <- [depth_128_map, depth_128_list_leaf] do
        assert_raise ArgumentError, ~r/depth/, fn ->
          VectorEntry.new([1.0, 0.0, 0.0, 0.0], id: "too_deep", metadata: too_deep)
        end

        assert {:error, "Invalid vector entry format"} =
                 RuvectorElixir.insert(db, %{
                   id: "too_deep",
                   vector: [1.0, 0.0, 0.0, 0.0],
                   metadata: too_deep
                 })

        assert {:error, "Invalid filter format"} =
                 RuvectorElixir.search(db, [1.0, 0.0, 0.0, 0.0], 5, filter: too_deep)
      end

      assert {:error, "Invalid filter format"} =
               RuvectorElixir.search(db, [1.0, 0.0, 0.0, 0.0], 5, filter: depth_128_kw)
    end

    test "rejects invalid open/3 options strictly", %{db: db, tmp_dir: tmp_dir} do
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
        RuvectorElixir.open(db_path, 4, quantization: %{type: "Scalar", subspaces: 4})
      end

      assert_raise ArgumentError, fn ->
        RuvectorElixir.open(db_path, 16, quantization: %{type: "SCALAR", subspaces: 8})
      end

      assert_raise ArgumentError, fn ->
        RuvectorElixir.open(db_path, 4, quantization: [type: "BINARY", k: 256])
      end

      assert_raise ArgumentError, fn ->
        RuvectorElixir.open(db_path, 4, quantization: %{type: "None", k: 128})
      end

      assert {:ok, scalar_db} =
               RuvectorElixir.open("memory://quant_scalar_mixed", 4,
                 quantization: %{type: "Scalar"}
               )

      assert {:ok, %{quantization: :scalar}} = RuvectorElixir.info(scalar_db)
      assert :ok = RuvectorElixir.close(scalar_db)

      assert {:ok, product_db} =
               RuvectorElixir.open("memory://quant_product_mixed", 4,
                 quantization: %{type: "Product", subspaces: 2}
               )

      assert {:ok, %{quantization: %{type: :product, subspaces: 2, k: 256}}} =
               RuvectorElixir.info(product_db)

      assert :ok = RuvectorElixir.close(product_db)

      assert_raise ArgumentError, fn ->
        RuvectorElixir.open(db_path, 4, metric: :invalid, metric: :cosine)
      end

      assert_raise ArgumentError, fn ->
        RuvectorElixir.open(db_path, 4, hnsw: [])
      end

      assert_raise ArgumentError, fn ->
        RuvectorElixir.open(db_path, 4, hnsw: %{})
      end

      assert_raise ArgumentError, fn ->
        RuvectorElixir.search(db, [1.0, 0.0, 0.0, 0.0], 5, ef_search: -1, ef_search: 10)
      end

      assert_raise ArgumentError, ~r/query_vector=/, fn ->
        RuvectorElixir.search(db, :not_a_list, 5)
      end

      assert_raise ArgumentError, ~r/db_ref=/, fn ->
        RuvectorElixir.search_detailed(:not_a_ref, [1.0, 0.0, 0.0, 0.0], 5)
      end

      assert_raise ArgumentError, fn ->
        RuvectorElixir.open(db_path, 4, quantization: %{type: :product, subspaces: 3, k: 256})
      end

      too_big_usize = Bitwise.bsl(1, :erlang.system_info(:wordsize) * 8)

      assert_raise ArgumentError, ~r/exceeds maximum platform integer size/, fn ->
        RuvectorElixir.open(db_path, too_big_usize)
      end

      for hnsw_key <- [:ef_construction, :ef_search, :max_elements] do
        assert_raise ArgumentError, ~r/exceeds maximum platform integer size/, fn ->
          RuvectorElixir.open(db_path, 4, hnsw: [{hnsw_key, too_big_usize}])
        end
      end

      assert_raise ArgumentError, ~r/exceeds maximum platform integer size/, fn ->
        RuvectorElixir.open(db_path, 4, quantization: [type: :product, subspaces: too_big_usize])
      end

      assert_raise ArgumentError, ~r/exceeds maximum platform integer size/, fn ->
        RuvectorElixir.open(db_path, 4,
          quantization: [type: :product, subspaces: 2, k: too_big_usize]
        )
      end

      assert_raise ArgumentError, ~r/exceeds maximum platform integer size/, fn ->
        RuvectorElixir.search(db, [1.0, 0.0, 0.0, 0.0], 5, ef_search: too_big_usize)
      end

      assert_raise ArgumentError, ~r/exceeds maximum platform integer size/, fn ->
        RuvectorElixir.search_detailed(db, [1.0, 0.0, 0.0, 0.0], 5, ef_search: too_big_usize)
      end

      assert_raise ArgumentError, ~r/between 1 and 10000000/, fn ->
        RuvectorElixir.open(db_path, 4, hnsw: [max_elements: 10_000_001])
      end

      assert_raise ArgumentError, ~r/between 1 and 10000000/, fn ->
        RuvectorElixir.open(db_path, 4, hnsw: [max_elements: Bitwise.bsl(1, 50)])
      end

      for ef_key <- [:ef_construction, :ef_search] do
        assert_raise ArgumentError, ~r/between 1 and 1000000/, fn ->
          RuvectorElixir.open(db_path, 4, hnsw: [{ef_key, 1_000_001}])
        end
      end

      # Direct Native.open_db_with_options enforces HNSW upper bounds and duplicate option checks
      assert {:error, reason} =
               RuvectorElixir.Native.open_db_with_options(db_path, 4, %{
                 "hnsw" => %{"max_elements" => 10_000_001}
               })

      assert String.contains?(reason, "max_elements")

      assert {:error, ef_reason} =
               RuvectorElixir.Native.open_db_with_options(db_path, 4, %{
                 "hnsw" => %{"ef_construction" => 1_000_001}
               })

      assert String.contains?(ef_reason, "ef_construction")

      for dup_opts <- [
            %{:hnsw => true, "hnsw" => false},
            %{:quantization => :scalar, "quantization" => :none},
            %{"hnsw" => %{:m => 16, "m" => 32}},
            %{"hnsw" => %{:ef_search => 32, "ef_search" => 64}},
            %{"hnsw" => %{:ef_construction => 50, "ef_construction" => 100}},
            %{"hnsw" => %{:max_elements => 100, "max_elements" => 200}}
          ] do
        assert {:error, dup_reason} =
                 RuvectorElixir.Native.open_db_with_options(db_path, 4, dup_opts)

        assert String.contains?(dup_reason, "Duplicate")
      end
    end

    test "search! and search_detailed! raise ArgumentError on invalid vectors/filters and Error on operational failures",
         %{db: db, tmp_dir: tmp_dir} do
      assert_raise ArgumentError, ~r/Invalid query vector/, fn ->
        RuvectorElixir.search!(db, ["bad"], 5)
      end

      assert_raise ArgumentError, ~r/Invalid query vector/, fn ->
        RuvectorElixir.search_detailed!(db, [1.0e100, 0.0, 0.0, 0.0], 5)
      end

      assert_raise ArgumentError, ~r/Invalid filter format/, fn ->
        RuvectorElixir.search!(db, [1.0, 0.0, 0.0, 0.0], 5, filter: %{"bad" => {:tuple}})
      end

      assert_raise ArgumentError, ~r/Invalid filter format/, fn ->
        RuvectorElixir.search_detailed!(db, [1.0, 0.0, 0.0, 0.0], 5, filter: %{"bad" => {:tuple}})
      end

      assert_raise Error, ~r/Dimension mismatch/, fn ->
        RuvectorElixir.search!(db, [1.0, 2.0], 5)
      end

      {:ok, closable} = RuvectorElixir.open(Path.join(tmp_dir, "bang_closed.rvf"), 4)
      :ok = RuvectorElixir.close(closable)

      assert_raise Error, ~r/closed/i, fn ->
        RuvectorElixir.search!(closable, [1.0, 0.0, 0.0, 0.0], 5)
      end

      assert_raise Error, ~r/closed/i, fn ->
        RuvectorElixir.search_detailed!(closable, [1.0, 0.0, 0.0, 0.0], 5)
      end
    end

    test "non-reference db_ref raises ArgumentError consistently across handle APIs" do
      calls = [
        {:close, [:not_a_ref]},
        {:close!, [:not_a_ref]},
        {:insert, [:not_a_ref, [1.0, 0.0, 0.0, 0.0]]},
        {:insert!, [:not_a_ref, [1.0, 0.0, 0.0, 0.0]]},
        {:insert_batch, [:not_a_ref, []]},
        {:insert_batch!, [:not_a_ref, []]},
        {:get, [:not_a_ref, "id"]},
        {:get!, [:not_a_ref, "id"]},
        {:get_struct, [:not_a_ref, "id"]},
        {:get_struct!, [:not_a_ref, "id"]},
        {:delete, [:not_a_ref, "id"]},
        {:delete!, [:not_a_ref, "id"]},
        {:len, [:not_a_ref]},
        {:len!, [:not_a_ref]},
        {:count, [:not_a_ref]},
        {:count!, [:not_a_ref]},
        {:empty?, [:not_a_ref]},
        {:keys, [:not_a_ref]},
        {:keys!, [:not_a_ref]},
        {:info, [:not_a_ref]},
        {:info!, [:not_a_ref]}
      ]

      for {fun, args} <- calls do
        assert_raise ArgumentError, ~r/expected db_ref/, fn ->
          apply(RuvectorElixir, fun, args)
        end
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

    test "memory:// database works without creating files on disk" do
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

      refute File.exists?(Path.join(File.cwd!(), "memory:"))
      assert :ok = RuvectorElixir.close(mem_db)
    end

    test "VectorEntry.new/2 and VectorEntry.from_map/1 enforce ID, vector, and metadata invariants" do
      assert_raise ArgumentError, fn ->
        VectorEntry.new([1.0, 2.0], id: 3.14)
      end

      assert_raise ArgumentError, fn ->
        VectorEntry.new([1.0, 2.0], id: true)
      end

      assert_raise ArgumentError, fn ->
        VectorEntry.new([1.0, 2.0], id: "")
      end

      assert_raise ArgumentError, fn ->
        VectorEntry.new([1.0, 2.0], id: :"")
      end

      assert_raise ArgumentError, fn ->
        VectorEntry.new([1.0, 2.0], id: Bitwise.bsl(1, 128))
      end

      assert_raise ArgumentError, fn ->
        VectorEntry.new([1.0, 2.0], id: -Bitwise.bsl(1, 127) - 1)
      end

      assert_raise ArgumentError, fn ->
        VectorEntry.new([])
      end

      assert_raise ArgumentError, fn ->
        VectorEntry.from_map(%{vector: []})
      end

      assert_raise ArgumentError, fn ->
        VectorEntry.new([1.0, 2.0], id: <<0xFF, 0xFE>>)
      end

      assert_raise ArgumentError, fn ->
        VectorEntry.new([1.0e100, 2.0])
      end

      assert_raise ArgumentError, fn ->
        VectorEntry.new([Integer.pow(10, 400)])
      end

      assert_raise ArgumentError, fn ->
        VectorEntry.new([-Integer.pow(10, 400)])
      end

      assert_raise ArgumentError, fn ->
        VectorEntry.from_map(%{vector: [Integer.pow(10, 400)]})
      end

      assert_raise ArgumentError, fn ->
        VectorEntry.new([340_282_356_779_733_661_637_539_395_458_142_568_448])
      end

      max_f32_int = 340_282_346_638_528_859_811_704_183_484_516_925_440
      max_entry = VectorEntry.new([max_f32_int])
      assert [val] = max_entry.vector
      assert is_float(val)

      assert_raise ArgumentError, fn ->
        VectorEntry.new([1.0, 2.0], metadata: ~D[2026-01-01])
      end

      assert_raise ArgumentError, ~r/Duplicate VectorEntry option/, fn ->
        VectorEntry.new([1.0, 2.0], %{:id => nil, "id" => 3.14})
      end

      assert_raise ArgumentError, fn ->
        VectorEntry.from_map(%{:id => nil, "id" => "doc_1", :vector => [1.0, 2.0]})
      end

      assert_raise ArgumentError, fn ->
        VectorEntry.from_map(%{:vector => nil, "vector" => [1.0, 2.0]})
      end

      assert_raise ArgumentError, fn ->
        VectorEntry.from_map(%{:vector => [1.0, 2.0], "vector" => "bad"})
      end

      assert_raise ArgumentError, fn ->
        VectorEntry.from_map(%{
          :metadata => nil,
          "metadata" => %{"a" => 1},
          :vector => [1.0, 2.0]
        })
      end

      assert_raise ArgumentError, fn ->
        VectorEntry.new([1.0 | 2.0])
      end

      assert_raise ArgumentError, fn ->
        VectorEntry.from_map(%{vector: [1.0 | 2.0]})
      end

      for bad_meta <- [
            %{"bad" => {:tuple}},
            %{"bad" => self()},
            %{"bad" => <<0xFF, 0xFE>>},
            %{:a => 1, "a" => 2},
            %{__struct__: "fake"},
            %{"nested" => ~D[2026-01-01]},
            %{"list" => [1 | 2]},
            %{"big" => Bitwise.bsl(1, 65)},
            %{1 => "int_key"},
            Enum.reduce(1..128, "leaf", fn _, acc -> %{"n" => acc} end),
            Enum.reduce(1..127, [], fn _, acc -> %{"n" => acc} end)
          ] do
        assert_raise ArgumentError, fn ->
          VectorEntry.new([1.0, 2.0], metadata: bad_meta)
        end

        assert_raise ArgumentError, fn ->
          VectorEntry.from_map(%{vector: [1.0, 2.0], metadata: bad_meta})
        end
      end

      entry = VectorEntry.from_map(%{id: 42, vector: [1.0, 2.0], metadata: nil})
      assert entry.id == "42"
      assert entry.vector == [1.0, 2.0]
    end

    test "handles large u128-range integers consistently and rejects improper lists across entry/search APIs",
         %{tmp_dir: tmp_dir} do
      big_int = 10_000_000_000_000_000_000
      big_entry = VectorEntry.new([big_int, 0, 0, 0], id: "big_struct")

      for metric <- [:cosine, :dot_product, :manhattan] do
        {:ok, handle} = RuvectorElixir.open("memory://big_int_#{metric}", 4, metric: metric)

        assert {:ok, "big_raw"} =
                 RuvectorElixir.insert(handle, %{id: "big_raw", vector: [big_int, 0, 0, 0]})

        assert {:ok, "big_struct"} = RuvectorElixir.insert(handle, big_entry)
        assert {:ok, ids} = RuvectorElixir.search(handle, [big_int, 0, 0, 0], 5)
        assert Enum.sort(ids) == ["big_raw", "big_struct"]

        assert :ok = RuvectorElixir.close(handle)
      end

      {:ok, db} = RuvectorElixir.open(Path.join(tmp_dir, "improper_lists.rvf"), 4)

      assert_raise ArgumentError, fn ->
        RuvectorElixir.search(db, [1.0 | 2.0], 5)
      end

      assert_raise ArgumentError, fn ->
        RuvectorElixir.search_detailed(db, [1.0 | 2.0], 5)
      end

      assert {:error, "Invalid vector entry format"} =
               RuvectorElixir.insert(db, [1.0 | 2.0])

      assert {:error, "Invalid vector entry format"} =
               RuvectorElixir.insert(db, %{
                 id: "bad_meta_list",
                 vector: [1.0, 0.0, 0.0, 0.0],
                 metadata: %{"bad" => [1 | 2]}
               })

      assert {:error, "Invalid vector entry in batch"} =
               RuvectorElixir.insert_batch(db, [[1.0 | 2.0]])

      assert :ok = RuvectorElixir.close(db)
    end

    test "rejects f32-distance-overflowing vectors before inserting or searching so DBs cannot be poisoned",
         %{tmp_dir: tmp_dir} do
      overflow_vec = [1.0e20, 1.0e20, 1.0e20, 1.0e20]
      manhattan_overflow = [3.0e38, 0.0, 0.0, 0.0]

      for {metric, bad_vec} <- [
            {:cosine, overflow_vec},
            {:euclidean, overflow_vec},
            {:dot_product, overflow_vec},
            {:manhattan, manhattan_overflow}
          ],
          hnsw_opt <- [false, true],
          backend <- [:memory, :disk] do
        path =
          case backend do
            :memory -> "memory://poison_#{metric}_#{hnsw_opt}"
            :disk -> Path.join(tmp_dir, "poison_#{metric}_#{hnsw_opt}.rvf")
          end

        {:ok, handle} = RuvectorElixir.open(path, 4, metric: metric, hnsw: hnsw_opt)

        assert {:ok, "good"} =
                 RuvectorElixir.insert(handle, %{id: "good", vector: [1.0, 0.0, 0.0, 0.0]})

        assert {:error, reason} =
                 RuvectorElixir.insert(handle, %{id: "bad", vector: bad_vec})

        assert String.contains?(reason, "magnitude")

        assert {:error, batch_reason} =
                 RuvectorElixir.insert_batch(handle, [
                   %{id: "other_good", vector: [0.0, 1.0, 0.0, 0.0]},
                   %{id: "bad_batch", vector: bad_vec}
                 ])

        assert String.contains?(batch_reason, "magnitude")
        assert {:ok, 1} = RuvectorElixir.len(handle)

        assert {:error, query_reason} = RuvectorElixir.search(handle, bad_vec, 5)
        assert String.contains?(query_reason, "magnitude")

        assert {:ok, ["good"]} = RuvectorElixir.search(handle, [1.0, 0.0, 0.0, 0.0], 5)

        assert :ok = RuvectorElixir.close(handle)
      end
    end

    test "HNSW search recovers live vectors past tombstones and selective filters", %{
      tmp_dir: tmp_dir
    } do
      targets = [
        {"memory://hnsw_tombstones", [metric: :euclidean, hnsw: [m: 16, ef_search: 16]]},
        {Path.join(tmp_dir, "hnsw_tombstones.rvf"),
         [metric: :euclidean, hnsw: [m: 16, ef_search: 16]]}
      ]

      for {path, opts} <- targets do
        assert {:ok, db} = RuvectorElixir.open(path, 4, opts)

        entries =
          for i <- 0..99 do
            x = if i < 90, do: i * 0.001, else: 10.0 + (i - 90) * 0.1

            %{
              id: "v_#{i}",
              vector: [x, 0.0, 0.0, 0.0],
              metadata: %{"keep" => i >= 90, "rare" => i in [95, 98]}
            }
          end

        assert {:ok, _} = RuvectorElixir.insert_batch(db, entries)

        for i <- 0..89 do
          assert {:ok, true} = RuvectorElixir.delete(db, "v_#{i}")
        end

        assert {:ok, 10} = RuvectorElixir.len(db)

        assert {:ok, top5} = RuvectorElixir.search(db, [0.0, 0.0, 0.0, 0.0], 5)
        assert top5 == ["v_90", "v_91", "v_92", "v_93", "v_94"]

        assert {:ok, all_live} = RuvectorElixir.search(db, [0.0, 0.0, 0.0, 0.0], 10)
        assert length(all_live) == 10
        assert hd(all_live) == "v_90"

        assert {:ok, filtered} =
                 RuvectorElixir.search_detailed(db, [0.0, 0.0, 0.0, 0.0], 5,
                   filter: %{"keep" => true},
                   ef_search: 8
                 )

        assert Enum.map(filtered, & &1.id) == ["v_90", "v_91", "v_92", "v_93", "v_94"]

        assert {:ok, rare} =
                 RuvectorElixir.search(db, [0.0, 0.0, 0.0, 0.0], 5, filter: %{"rare" => true})

        assert rare == ["v_95", "v_98"]

        assert :ok = RuvectorElixir.close(db)
      end
    end

    test "insert_batch deduplicates intra-batch duplicate IDs without ghost search results", %{
      tmp_dir: tmp_dir
    } do
      targets = [
        {"memory://dedupe_flat", [metric: :euclidean, hnsw: false]},
        {"memory://dedupe_hnsw", [metric: :euclidean, hnsw: [m: 16, ef_search: 32]]},
        {Path.join(tmp_dir, "dedupe_flat.rvf"), [metric: :euclidean, hnsw: false]},
        {Path.join(tmp_dir, "dedupe_hnsw.rvf"),
         [metric: :euclidean, hnsw: [m: 16, ef_search: 32]]}
      ]

      v1 = [1.0, 0.0, 0.0, 0.0]
      v2 = [0.0, 0.0, 1.0, 0.0]
      v_other = [0.0, 1.0, 0.0, 0.0]

      for {path, opts} <- targets do
        assert {:ok, db} = RuvectorElixir.open(path, 4, opts)

        # Invalid earlier overwritten entry still rejects the batch atomically
        assert {:error, reason} =
                 RuvectorElixir.insert_batch(db, [
                   %{id: "dup", vector: [1.0, 2.0]},
                   %{id: "dup", vector: v2}
                 ])

        assert String.contains?(reason, "Dimension") or String.contains?(reason, "mismatch")

        assert {:ok, ["other", "dup"]} =
                 RuvectorElixir.insert_batch(db, [
                   %{id: "dup", vector: v1, metadata: %{"rev" => 1}},
                   %{id: "other", vector: v_other, metadata: %{"rev" => 0}},
                   %{id: "dup", vector: v2, metadata: %{"rev" => 2}}
                 ])

        assert {:ok, 2} = RuvectorElixir.len(db)
        assert {:ok, fetched} = RuvectorElixir.get(db, "dup")
        assert fetched.vector == v2
        assert fetched.metadata == %{"rev" => 2}

        assert {:ok, ids} = RuvectorElixir.search(db, v1, 5)
        assert Enum.sort(ids) == ["dup", "other"]
        assert Enum.count(ids, &(&1 == "dup")) == 1

        assert {:ok, detailed} = RuvectorElixir.search_detailed(db, v1, 5)
        dup_res = Enum.find(detailed, &(&1.id == "dup"))
        assert dup_res.vector == v2
        assert dup_res.metadata == %{"rev" => 2}
        assert_in_delta dup_res.score, :math.sqrt(2.0), 0.01

        assert {:ok, true} = RuvectorElixir.delete(db, "dup")
        assert {:ok, 1} = RuvectorElixir.len(db)
        assert {:ok, ["other"]} = RuvectorElixir.search(db, v1, 5)

        assert :ok = RuvectorElixir.close(db)
      end
    end

    test "persistent Flat and HNSW search reflects updated vectors after single and batch upserts",
         %{tmp_dir: tmp_dir} do
      for hnsw_opt <- [false, [m: 16, ef_search: 32]] do
        db_path = Path.join(tmp_dir, "upsert_search_#{inspect(hnsw_opt != false)}.rvf")
        assert {:ok, db} = RuvectorElixir.open(db_path, 4, metric: :euclidean, hnsw: hnsw_opt)

        assert {:ok, "dup_id"} =
                 RuvectorElixir.insert(db, %{
                   id: "dup_id",
                   vector: [10.0, 0.0, 0.0, 0.0],
                   metadata: %{"rev" => 1}
                 })

        assert {:ok, "other"} =
                 RuvectorElixir.insert(db, %{
                   id: "other",
                   vector: [9.0, 0.0, 0.0, 0.0],
                   metadata: %{"rev" => 0}
                 })

        # Single-insert upsert moves "dup_id" away from [10, 0, 0, 0] to [0, 10, 0, 0]
        assert {:ok, "dup_id"} =
                 RuvectorElixir.insert(db, %{
                   id: "dup_id",
                   vector: [0.0, 10.0, 0.0, 0.0],
                   metadata: %{"rev" => 2}
                 })

        assert {:ok, [first, second]} =
                 RuvectorElixir.search_detailed(db, [10.0, 0.0, 0.0, 0.0], 2)

        assert first.id == "other"
        assert_in_delta first.score, 1.0, 0.01
        assert second.id == "dup_id"
        assert second.vector == [0.0, 10.0, 0.0, 0.0]
        assert second.metadata == %{"rev" => 2}
        assert_in_delta second.score, :math.sqrt(200.0), 0.01

        # Batch upsert moves "dup_id" right onto [10, 0, 0, 0]
        assert {:ok, ["dup_id"]} =
                 RuvectorElixir.insert_batch(db, [
                   %{id: "dup_id", vector: [10.0, 0.0, 0.0, 0.0], metadata: %{"rev" => 3}}
                 ])

        assert {:ok, [now_first, now_second]} =
                 RuvectorElixir.search_detailed(db, [10.0, 0.0, 0.0, 0.0], 2)

        assert now_first.id == "dup_id"
        assert_in_delta now_first.score, 0.0, 0.01
        assert now_first.metadata == %{"rev" => 3}
        assert now_second.id == "other"
        assert_in_delta now_second.score, 1.0, 0.01

        assert :ok = RuvectorElixir.close(db)
      end
    end

    test "with_open/3,4 and GC reaper release persistent DB handles safely", %{tmp_dir: tmp_dir} do
      db_path = Path.join(tmp_dir, "with_open.rvf")

      assert {:ok, {:inserted, "w1"}} =
               RuvectorElixir.with_open(db_path, 4, [metric: :euclidean], fn db ->
                 {:ok, id} = RuvectorElixir.insert(db, %{id: "w1", vector: [1.0, 2.0, 3.0, 4.0]})
                 {:inserted, id}
               end)

      # Reopening immediately succeeds because with_open closed the handle
      assert {:ok, 1} =
               RuvectorElixir.with_open(db_path, 4, fn db ->
                 {:ok, total} = RuvectorElixir.len(db)
                 total
               end)

      # Exception inside callback still closes the handle cleanly
      assert_raise RuntimeError, "boom", fn ->
        RuvectorElixir.with_open!(db_path, 4, fn _db ->
          raise "boom"
        end)
      end

      assert "w1" ==
               RuvectorElixir.with_open!(db_path, 4, fn db ->
                 assert {:ok, entry} = RuvectorElixir.get(db, "w1")
                 entry.id
               end)

      assert_raise ArgumentError, fn ->
        RuvectorElixir.with_open(db_path, 4, :not_a_fun)
      end

      assert_raise ArgumentError, fn ->
        RuvectorElixir.with_open!(db_path, 4, [], :not_a_fun)
      end

      # Unclosed persistent handle collected by GC is offloaded to background reaper
      gc_path = Path.join(tmp_dir, "gc_reaper.rvf")

      task =
        Task.async(fn ->
          {:ok, leaked_db} = RuvectorElixir.open(gc_path, 4)
          {:ok, _} = RuvectorElixir.insert(leaked_db, %{id: "gc1", vector: [1.0, 0.0, 0.0, 0.0]})
          :ok
        end)

      assert :ok = Task.await(task)
      :erlang.garbage_collect()

      assert {:ok, entry} = reopen_eventually(gc_path, 4, 25)
      assert entry.id == "gc1"
    end
  end

  defp reopen_eventually(path, dims, attempts_left) do
    case RuvectorElixir.with_open(path, dims, fn db -> RuvectorElixir.get(db, "gc1") end) do
      {:ok, {:ok, entry}} ->
        {:ok, entry}

      {:error, _reason} when attempts_left > 1 ->
        Process.sleep(20)
        :erlang.garbage_collect()
        reopen_eventually(path, dims, attempts_left - 1)

      other ->
        other
    end
  end
end
