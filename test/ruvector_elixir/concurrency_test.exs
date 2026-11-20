defmodule RuvectorElixir.ConcurrencyTest do
  use ExUnit.Case, async: true

  @moduletag :tmp_dir

  setup %{tmp_dir: tmp_dir} do
    db_path = Path.join(tmp_dir, "ruvector_concurrent.rvf")
    {:ok, db} = RuvectorElixir.open(db_path, 32)
    {:ok, db: db, db_path: db_path, tmp_dir: tmp_dir}
  end

  describe "concurrent operations (matching upstream concurrent_tests.rs)" do
    test "concurrent reads across multiple BEAM tasks", %{db: db} do
      for i <- 0..99 do
        vec = for(j <- 0..31, do: (i + j) * 0.1)
        assert {:ok, _} = RuvectorElixir.insert(db, %{id: "vec_#{i}", vector: vec})
      end

      assert {:ok, 100} = RuvectorElixir.len(db)

      tasks =
        for task_id <- 0..9 do
          Task.async(fn ->
            for i <- 0..49 do
              id = "vec_#{rem(task_id * 10 + i, 100)}"
              assert {:ok, entry} = RuvectorElixir.get(db, id)
              assert entry.id == id
              assert length(entry.vector) == 32
            end
          end)
        end

      Task.await_many(tasks, 10_000)
    end

    test "concurrent writes with no ID collision across multiple BEAM tasks", %{db: db} do
      num_tasks = 10
      vectors_per_task = 20

      tasks =
        for task_id <- 0..(num_tasks - 1) do
          Task.async(fn ->
            for i <- 0..(vectors_per_task - 1) do
              id = "task_#{task_id}_#{i}"
              vec = List.duplicate(task_id * 1.0, 32)
              assert {:ok, ^id} = RuvectorElixir.insert(db, %{id: id, vector: vec})
            end
          end)
        end

      Task.await_many(tasks, 10_000)

      expected_total = num_tasks * vectors_per_task
      assert {:ok, ^expected_total} = RuvectorElixir.len(db)
    end

    test "concurrent reads and writes simultaneously", %{db: db} do
      for i <- 0..49 do
        assert {:ok, _} =
                 RuvectorElixir.insert(db, %{id: "init_#{i}", vector: List.duplicate(i * 0.1, 32)})
      end

      writers =
        for task_id <- 0..4 do
          Task.async(fn ->
            for i <- 0..19 do
              id = "writer_#{task_id}_#{i}"
              vec = List.duplicate(task_id * 1.0, 32)
              assert {:ok, ^id} = RuvectorElixir.insert(db, %{id: id, vector: vec})
            end
          end)
        end

      readers =
        for task_id <- 0..4 do
          Task.async(fn ->
            for i <- 0..29 do
              id = "init_#{rem(task_id * 10 + i, 50)}"
              assert {:ok, entry} = RuvectorElixir.get(db, id)
              assert entry.id == id
            end
          end)
        end

      Task.await_many(writers ++ readers, 10_000)
      assert {:ok, 150} = RuvectorElixir.len(db)
    end

    test "concurrent ANN searches across multiple BEAM tasks", %{db: db} do
      for i <- 0..99 do
        vec = for(j <- 0..31, do: (i + j) * 0.05)
        assert {:ok, _} = RuvectorElixir.insert(db, %{id: "ann_#{i}", vector: vec})
      end

      tasks =
        for task_id <- 0..9 do
          Task.async(fn ->
            for _i <- 0..19 do
              query = List.duplicate(task_id * 0.1, 32)
              assert {:ok, results} = RuvectorElixir.search(db, query, 5)
              assert is_list(results)
              assert length(results) == 5
            end
          end)
        end

      Task.await_many(tasks, 10_000)
    end

    test "concurrent operations on separate database handle", %{tmp_dir: tmp_dir} do
      db_path = Path.join(tmp_dir, "ruvector_concurrent_sep.rvf")
      {:ok, sep_db} = RuvectorElixir.open(db_path, 16)

      tasks =
        for task_id <- 0..7 do
          Task.async(fn ->
            for i <- 0..14 do
              id = "mem_#{task_id}_#{i}"
              vec = List.duplicate(i * 0.2, 16)
              assert {:ok, ^id} = RuvectorElixir.insert(sep_db, %{id: id, vector: vec})
              assert {:ok, entry} = RuvectorElixir.get(sep_db, id)
              assert entry.id == id
            end
          end)
        end

      Task.await_many(tasks, 10_000)
      assert {:ok, 120} = RuvectorElixir.len(sep_db)
    end

    test "concurrent HNSW reads, writes, deletes, upserts, and searches", %{tmp_dir: tmp_dir} do
      db_path = Path.join(tmp_dir, "ruvector_hnsw_concurrent.rvf")

      assert {:ok, hnsw_db} =
               RuvectorElixir.open(db_path, 32,
                 metric: :cosine,
                 hnsw: [m: 16, ef_construction: 64, ef_search: 64, max_elements: 2_000]
               )

      for i <- 0..49 do
        vec = for j <- 0..31, do: (i + j + 1) * 0.05
        assert {:ok, _} = RuvectorElixir.insert(hnsw_db, %{id: "h_#{i}", vector: vec})
      end

      writers =
        for t <- 0..3 do
          Task.async(fn ->
            for i <- 0..9 do
              id = "hw_#{t}_#{i}"
              vec = for j <- 0..31, do: (t * 10 + i + j + 1) * 0.02
              assert {:ok, ^id} = RuvectorElixir.insert(hnsw_db, %{id: id, vector: vec})
              # Upsert an existing ID
              assert {:ok, ^id} = RuvectorElixir.insert(hnsw_db, %{id: id, vector: vec})
            end
          end)
        end

      deleters =
        Task.async(fn ->
          for i <- 0..9 do
            assert {:ok, true} = RuvectorElixir.delete(hnsw_db, "h_#{i}")
          end
        end)

      searchers =
        for t <- 0..3 do
          Task.async(fn ->
            for _ <- 0..9 do
              query = for j <- 0..31, do: (t + j + 1) * 0.05
              assert {:ok, ids} = RuvectorElixir.search(hnsw_db, query, 5)
              assert length(ids) == 5
            end
          end)
        end

      Task.await_many(writers ++ [deleters] ++ searchers, 10_000)
      assert {:ok, 80} = RuvectorElixir.len(hnsw_db)
    end
  end
end
