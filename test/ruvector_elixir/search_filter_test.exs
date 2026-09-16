defmodule RuvectorElixir.SearchFilterTest do
  use ExUnit.Case, async: true

  setup do
    db_path =
      Path.join(System.tmp_dir!(), "ruvector_filter_#{System.unique_integer([:positive])}.rvf")

    on_exit(fn -> File.rm_rf(db_path) end)

    {:ok, db} = RuvectorElixir.open(db_path, 16)

    # Insert 50 vectors with varied metadata
    entries =
      for i <- 0..49 do
        %{
          id: "vec_#{i}",
          vector: for(j <- 0..15, do: (i + j) * 0.1),
          metadata: %{
            "category" => rem(i, 3),
            "decade" => div(i, 10),
            "active" => rem(i, 2) == 0,
            "label" => "tag_#{i}"
          }
        }
      end

    {:ok, _} = RuvectorElixir.insert_batch(db, entries)

    {:ok, db: db}
  end

  describe "metadata filtering (matching upstream test_complex_metadata_filtering)" do
    test "filter by integer field (category = 0)", %{db: db} do
      query = for j <- 0..15, do: j * 0.1
      filter = %{"category" => 0}

      {:ok, results} = RuvectorElixir.search_detailed(db, query, filter, 100)
      assert length(results) > 0

      for res <- results do
        assert res.metadata["category"] == 0
      end
    end

    test "filter by another field (decade = 2)", %{db: db} do
      query = for j <- 0..15, do: j * 0.1
      filter = %{"decade" => 2}

      {:ok, results} = RuvectorElixir.search_detailed(db, query, filter, 100)
      # decade 2 should correspond to i in 20..29 -> 10 vectors
      assert length(results) == 10

      for res <- results do
        assert res.metadata["decade"] == 2
      end
    end

    test "filter with atom keys is supported", %{db: db} do
      query = for j <- 0..15, do: j * 0.1
      filter = %{category: 1}

      {:ok, results} = RuvectorElixir.search_detailed(db, query, filter, 50)
      assert length(results) > 0

      for res <- results do
        assert res.metadata["category"] == 1
      end
    end

    test "filter with boolean field", %{db: db} do
      query = for j <- 0..15, do: j * 0.1
      filter = %{"active" => true}

      {:ok, results} = RuvectorElixir.search_detailed(db, query, filter, 100)
      assert length(results) == 25

      for res <- results do
        assert res.metadata["active"] == true
      end
    end

    test "filter matching no records returns empty list", %{db: db} do
      query = for j <- 0..15, do: j * 0.1
      filter = %{"category" => 999}

      results = RuvectorElixir.search(db, query, filter, 10)
      assert results == []

      {:ok, detailed} = RuvectorElixir.search_detailed(db, query, filter, 10)
      assert detailed == []
    end

    test "nil or empty filter returns unfiltered results", %{db: db} do
      query = for j <- 0..15, do: j * 0.1

      all_results = RuvectorElixir.search(db, query, 10)
      assert length(all_results) == 10

      with_nil = RuvectorElixir.search(db, query, nil, 10)
      assert with_nil == all_results

      with_empty_map = RuvectorElixir.search(db, query, %{}, 10)
      assert with_empty_map == all_results
    end

    test "k larger than total elements returns all available elements", %{db: db} do
      query = for j <- 0..15, do: j * 0.1
      results = RuvectorElixir.search(db, query, 500)
      assert length(results) == 50
    end

    test "search with k = 1 returns exactly the single closest element", %{db: db} do
      query = for j <- 0..15, do: j * 0.1
      results = RuvectorElixir.search(db, query, 1)
      assert length(results) == 1
      assert hd(results) == "vec_0"
    end

    test "empty filter %{} does not drop vectors with nil metadata" do
      nil_db_path =
        Path.join(
          System.tmp_dir!(),
          "ruvector_nil_meta_#{System.unique_integer([:positive])}.rvf"
        )

      on_exit(fn -> File.rm_rf(nil_db_path) end)

      {:ok, db} = RuvectorElixir.open(nil_db_path, 3)
      {:ok, "v_no_meta"} = RuvectorElixir.insert(db, %{id: "v_no_meta", vector: [1.0, 2.0, 3.0]})

      # Searching with %{} must find the vector, not drop it!
      assert RuvectorElixir.search(db, [1.0, 2.0, 3.0], %{}, 5) == ["v_no_meta"]
      assert {:ok, [res]} = RuvectorElixir.search_detailed(db, [1.0, 2.0, 3.0], %{}, 5)
      assert res.id == "v_no_meta"
    end

    test "filter with keyword list [category: 0] is supported", %{db: db} do
      query = for j <- 0..15, do: j * 0.1
      results = RuvectorElixir.search(db, query, [category: 0], 10)
      assert length(results) > 0

      {:ok, detailed} = RuvectorElixir.search_detailed(db, query, [category: 0], 10)

      for r <- detailed do
        assert r.metadata["category"] == 0
      end
    end

    test "search! and search_detailed! return direct results and raise on error", %{db: db} do
      query = for j <- 0..15, do: j * 0.1
      results = RuvectorElixir.search!(db, query, 5)
      assert length(results) == 5

      detailed = RuvectorElixir.search_detailed!(db, query, [category: 1], 5)
      assert length(detailed) > 0

      # Dimension mismatch raises ArgumentError
      assert_raise ArgumentError, fn ->
        RuvectorElixir.search!(db, [1.0, 2.0], 5)
      end

      assert_raise ArgumentError, fn ->
        RuvectorElixir.search_detailed!(db, [1.0, 2.0], 5)
      end
    end
  end

  describe "scale and batch stress tests" do
    test "insert and search 500 vectors" do
      db_path =
        Path.join(System.tmp_dir!(), "ruvector_scale_#{System.unique_integer([:positive])}.rvf")

      on_exit(fn -> File.rm_rf(db_path) end)

      {:ok, db} = RuvectorElixir.open(db_path, 32)

      batch =
        for i <- 0..499 do
          %{
            id: "s_#{i}",
            vector: for(j <- 0..31, do: (i * 32 + j) * 0.001),
            metadata: %{"group" => rem(i, 5)}
          }
        end

      assert {:ok, ids} = RuvectorElixir.insert_batch(db, batch)
      assert length(ids) == 500
      assert {:ok, 500} = RuvectorElixir.len(db)

      query = for j <- 0..31, do: j * 0.001
      results = RuvectorElixir.search(db, query, %{"group" => 0}, 100)
      assert length(results) == 20
      assert hd(results) == "s_0"
    end
  end
end
