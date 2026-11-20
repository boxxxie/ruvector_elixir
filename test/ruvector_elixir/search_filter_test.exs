defmodule RuvectorElixir.SearchFilterTest do
  use ExUnit.Case, async: true

  alias RuvectorElixir.Error

  @moduletag :tmp_dir

  setup %{tmp_dir: tmp_dir} do
    db_path = Path.join(tmp_dir, "ruvector_filter.rvf")
    {:ok, db} = RuvectorElixir.open(db_path, 16)

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

      assert {:ok, results} = RuvectorElixir.search_detailed(db, query, 100, filter: filter)
      assert length(results) > 0

      for res <- results do
        assert res.metadata["category"] == 0
      end
    end

    test "filter by another field (decade = 2)", %{db: db} do
      query = for j <- 0..15, do: j * 0.1
      filter = %{"decade" => 2}

      assert {:ok, results} = RuvectorElixir.search_detailed(db, query, 100, filter: filter)
      assert length(results) == 10

      for res <- results do
        assert res.metadata["decade"] == 2
      end
    end

    test "filter with atom keys is supported", %{db: db} do
      query = for j <- 0..15, do: j * 0.1
      filter = %{category: 1}

      assert {:ok, results} = RuvectorElixir.search_detailed(db, query, 50, filter: filter)
      assert length(results) > 0

      for res <- results do
        assert res.metadata["category"] == 1
      end
    end

    test "filter with boolean field", %{db: db} do
      query = for j <- 0..15, do: j * 0.1
      filter = %{"active" => true}

      assert {:ok, results} = RuvectorElixir.search_detailed(db, query, 100, filter: filter)
      assert length(results) == 25

      for res <- results do
        assert res.metadata["active"] == true
      end
    end

    test "filter matching no records returns empty list", %{db: db} do
      query = for j <- 0..15, do: j * 0.1
      filter = %{"category" => 999}

      assert {:ok, []} = RuvectorElixir.search(db, query, 10, filter: filter)
      assert {:ok, []} = RuvectorElixir.search_detailed(db, query, 10, filter: filter)
    end

    test "nil or empty filter returns unfiltered results", %{db: db} do
      query = for j <- 0..15, do: j * 0.1

      assert {:ok, all_results} = RuvectorElixir.search(db, query, 10)
      assert length(all_results) == 10

      assert {:ok, ^all_results} = RuvectorElixir.search(db, query, 10, filter: nil)
      assert {:ok, ^all_results} = RuvectorElixir.search(db, query, 10, filter: %{})
    end

    test "k larger than total elements returns all available elements", %{db: db} do
      query = for j <- 0..15, do: j * 0.1
      assert {:ok, results} = RuvectorElixir.search(db, query, 500)
      assert length(results) == 50
    end

    test "search with k = 1 returns exactly the single closest element", %{db: db} do
      query = for j <- 0..15, do: j * 0.1
      assert {:ok, ["vec_0"]} = RuvectorElixir.search(db, query, 1)
    end

    @tag :tmp_dir
    test "empty filter %{} does not drop vectors with nil metadata", %{tmp_dir: tmp_dir} do
      nil_db_path = Path.join(tmp_dir, "ruvector_nil_meta.rvf")

      assert {:ok, db} = RuvectorElixir.open(nil_db_path, 3)

      assert {:ok, "v_no_meta"} =
               RuvectorElixir.insert(db, %{id: "v_no_meta", vector: [1.0, 2.0, 3.0]})

      assert {:ok, ["v_no_meta"]} = RuvectorElixir.search(db, [1.0, 2.0, 3.0], 5, filter: %{})

      assert {:ok, [res]} =
               RuvectorElixir.search_detailed(db, [1.0, 2.0, 3.0], 5, filter: %{})

      assert res.id == "v_no_meta"
    end

    test "filter with keyword list [category: 0] and first-key-wins semantics", %{db: db} do
      query = for j <- 0..15, do: j * 0.1

      assert {:ok, results} = RuvectorElixir.search(db, query, 10, filter: [category: 0])
      assert length(results) > 0

      assert {:ok, detailed} =
               RuvectorElixir.search_detailed(db, query, 10, filter: [category: 0])

      for r <- detailed do
        assert r.metadata["category"] == 0
      end

      assert {:ok, dup_results} =
               RuvectorElixir.search_detailed(db, query, 10, filter: [category: 1, category: 2])

      assert length(dup_results) > 0

      for r <- dup_results do
        assert r.metadata["category"] == 1
      end
    end

    test "numeric equality between integer and float metadata values", %{db: db} do
      assert {:ok, "float_ver"} =
               RuvectorElixir.insert(db, %{
                 id: "float_ver",
                 vector: List.duplicate(0.0, 16),
                 metadata: %{"version" => 1.0}
               })

      assert {:ok, ["float_ver"]} =
               RuvectorElixir.search(db, List.duplicate(0.0, 16), 5, filter: %{"version" => 1})
    end

    test "search! and search_detailed! return direct results and raise RuvectorElixir.Error on operational error",
         %{db: db} do
      query = for j <- 0..15, do: j * 0.1
      results = RuvectorElixir.search!(db, query, 5)
      assert length(results) == 5

      detailed = RuvectorElixir.search_detailed!(db, query, 5, filter: [category: 1])
      assert length(detailed) > 0

      assert_raise Error, fn ->
        RuvectorElixir.search!(db, [1.0, 2.0], 5)
      end

      assert_raise Error, fn ->
        RuvectorElixir.search_detailed!(db, [1.0, 2.0], 5)
      end
    end
  end

  describe "scale and batch stress tests" do
    @tag :tmp_dir
    test "insert and search 500 vectors", %{tmp_dir: tmp_dir} do
      db_path = Path.join(tmp_dir, "ruvector_scale.rvf")
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
      assert {:ok, results} = RuvectorElixir.search(db, query, 100, filter: %{"group" => 0})
      assert length(results) == 100
      assert hd(results) == "s_0"
    end
  end
end
