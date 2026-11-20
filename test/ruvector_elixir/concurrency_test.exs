defmodule RuvectorElixir.ConcurrencyTest do
  use ExUnit.Case, async: true

  @moduletag :tmp_dir

  setup %{tmp_dir: tmp_dir} do
    db_path = Path.join(tmp_dir, "ruvector_concurrent.rvf")
    {:ok, db} = RuvectorElixir.open(db_path, 32)
    on_exit(fn -> RuvectorElixir.close(db) end)
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
        vec = for j <- 0..31, do: :math.sin((i + 1) * (j + 1) * 0.13) + 1.0
        assert {:ok, _} = RuvectorElixir.insert(db, %{id: "ann_#{i}", vector: vec})
      end

      tasks =
        for task_id <- 0..9 do
          Task.async(fn ->
            query = for j <- 0..31, do: :math.sin((task_id * 10 + 1) * (j + 1) * 0.13) + 1.0

            for _i <- 0..19 do
              assert {:ok, results} = RuvectorElixir.search(db, query, 5)
              assert is_list(results)
              assert length(results) == 5
              assert hd(results) == "ann_#{task_id * 10}"
            end
          end)
        end

      Task.await_many(tasks, 10_000)
    end

    test "concurrent operations on separate database handle", %{tmp_dir: tmp_dir} do
      db_path = Path.join(tmp_dir, "ruvector_concurrent_sep.rvf")
      {:ok, sep_db} = RuvectorElixir.open(db_path, 16)
      on_exit(fn -> RuvectorElixir.close(sep_db) end)

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

      on_exit(fn -> RuvectorElixir.close(hnsw_db) end)

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

  describe "memory:// upserts, deletes, ef_search, and concurrency (flat & HNSW)" do
    for hnsw_opt <- [false, [m: 16, ef_construction: 64, ef_search: 32, max_elements: 1_000]] do
      test "upsert, delete, ef_search, and concurrent ops on memory:// with hnsw=#{inspect(hnsw_opt)}" do
        hnsw_mode = unquote(Macro.escape(hnsw_opt))

        assert {:ok, db} =
                 RuvectorElixir.open("memory://mem_suite", 8, metric: :euclidean, hnsw: hnsw_mode)

        # Upsert replaces stored vector and indexed vector atomically
        assert {:ok, "u1"} =
                 RuvectorElixir.insert(db, %{
                   id: "u1",
                   vector: [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   metadata: %{"rev" => 1}
                 })

        assert {:ok, "u1"} =
                 RuvectorElixir.insert(db, %{
                   id: "u1",
                   vector: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   metadata: %{"rev" => 2}
                 })

        assert {:ok, 1} = RuvectorElixir.len(db)

        assert {:ok, [res]} =
                 RuvectorElixir.search_detailed(db, List.duplicate(0.0, 8), 1, ef_search: 64)

        assert res.id == "u1"
        assert res.metadata == %{"rev" => 2}
        assert_in_delta res.score, 0.0, 0.001

        # Delete removes from both store and index
        assert {:ok, true} = RuvectorElixir.delete(db, "u1")
        assert {:ok, []} = RuvectorElixir.search(db, List.duplicate(0.0, 8), 5)

        # Concurrent reads, writes, upserts, and deletes on memory://
        tasks =
          for t <- 0..7 do
            Task.async(fn ->
              for i <- 0..9 do
                id = "m_#{t}_#{i}"
                vec = for j <- 0..7, do: (t + i + j) * 0.1
                assert {:ok, ^id} = RuvectorElixir.insert(db, %{id: id, vector: vec})
                assert {:ok, ^id} = RuvectorElixir.insert(db, %{id: id, vector: vec})
                assert {:ok, entry} = RuvectorElixir.get(db, id)
                assert entry.id == id
              end

              assert {:ok, true} = RuvectorElixir.delete(db, "m_#{t}_0")
              assert {:ok, ids} = RuvectorElixir.search(db, List.duplicate(0.5, 8), 5)
              assert length(ids) > 0
            end)
          end

        Task.await_many(tasks, 10_000)
        assert {:ok, 72} = RuvectorElixir.len(db)
        assert :ok = RuvectorElixir.close(db)
      end
    end

    test "close/1 completes promptly under continuous DirtyCpu search load", %{tmp_dir: tmp_dir} do
      db_path = Path.join(tmp_dir, "rwlock_fairness.rvf")

      assert {:ok, db} =
               RuvectorElixir.open(db_path, 16, hnsw: [m: 16, ef_search: 64, max_elements: 1_000])

      entries =
        for i <- 0..199 do
          %{id: "f_#{i}", vector: for(j <- 0..15, do: (i + j) * 0.01)}
        end

      assert {:ok, _} = RuvectorElixir.insert_batch(db, entries)

      query = List.duplicate(0.5, 16)
      parent = self()

      searchers =
        for _ <- 1..6 do
          Task.async(fn ->
            send(parent, :searcher_started)

            Enum.reduce_while(1..500, :ok, fn _, _acc ->
              case RuvectorElixir.search(db, query, 10) do
                {:ok, _ids} -> {:cont, :ok}
                {:error, :closed} -> {:halt, :closed}
              end
            end)
          end)
        end

      for _ <- 1..6 do
        assert_receive :searcher_started, 2_000
      end

      assert :ok = RuvectorElixir.close(db)
      results = Task.await_many(searchers, 5_000)
      assert Enum.all?(results, &(&1 in [:ok, :closed]))
    end
  end
end
