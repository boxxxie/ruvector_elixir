defmodule RuvectorElixir.StorageTest do
  use ExUnit.Case, async: true

  alias RuvectorElixir.{Error, VectorEntry}

  @moduletag :tmp_dir

  setup %{tmp_dir: tmp_dir} do
    db_path = Path.join(tmp_dir, "ruvector_storage.rvf")
    {:ok, db} = RuvectorElixir.open(db_path, 3)
    on_exit(fn -> RuvectorElixir.close(db) end)
    {:ok, db: db, db_path: db_path}
  end

  describe "storage operations (matching upstream storage_tests)" do
    test "insert with explicit id", %{db: db} do
      entry = %{
        id: "explicit_id",
        vector: [1.0, 2.0, 3.0],
        metadata: nil
      }

      assert {:ok, "explicit_id"} = RuvectorElixir.insert(db, entry)
      assert {:ok, retrieved} = RuvectorElixir.get(db, "explicit_id")
      assert retrieved.id == "explicit_id"
      assert retrieved.vector == [1.0, 2.0, 3.0]
    end

    test "insert with VectorEntry struct", %{db: db} do
      entry = VectorEntry.new([1.0, 2.0, 3.0], id: "struct_entry", metadata: %{"tag" => "v1"})

      assert {:ok, "struct_entry"} = RuvectorElixir.insert(db, entry)
      assert {:ok, retrieved} = RuvectorElixir.get(db, "struct_entry")
      assert retrieved.id == "struct_entry"
      assert retrieved.vector == [1.0, 2.0, 3.0]
      assert retrieved.metadata == %{"tag" => "v1"}
    end

    test "insert auto generates id when omitted or nil", %{db: db} do
      assert {:ok, id1} = RuvectorElixir.insert(db, %{vector: [1.0, 2.0, 3.0]})
      assert is_binary(id1)
      assert String.contains?(id1, "-")

      assert {:ok, id2} = RuvectorElixir.insert(db, [4.0, 5.0, 6.0])
      assert is_binary(id2)
      assert String.contains?(id2, "-")

      assert id1 != id2
      assert {:ok, 2} = RuvectorElixir.len(db)
    end

    test "insert with rich metadata", %{db: db} do
      metadata = %{
        "string" => "hello",
        "int" => 42,
        "float" => 3.14,
        "bool" => true,
        "list" => ["a", "b", "c"],
        "nested" => %{"inner" => "world"}
      }

      entry = %{
        id: "meta_test",
        vector: [1.0, 2.0, 3.0],
        metadata: metadata
      }

      assert {:ok, "meta_test"} = RuvectorElixir.insert(db, entry)
      assert {:ok, retrieved} = RuvectorElixir.get(db, "meta_test")
      assert retrieved.metadata == metadata
    end

    test "dimension mismatch error on insert", %{db: db} do
      entry = %{
        id: "wrong_dim",
        vector: [1.0, 2.0]
      }

      assert {:error, reason} = RuvectorElixir.insert(db, entry)
      assert String.contains?(reason, "mismatch") or String.contains?(reason, "Dimension")
    end

    test "get nonexistent returns {:error, :not_found}", %{db: db} do
      assert {:error, :not_found} = RuvectorElixir.get(db, "nonexistent")
    end

    test "delete existing entry", %{db: db} do
      assert {:ok, "del_me"} = RuvectorElixir.insert(db, %{id: "del_me", vector: [1.0, 2.0, 3.0]})
      assert {:ok, 1} = RuvectorElixir.count(db)

      assert {:ok, true} = RuvectorElixir.delete(db, "del_me")
      assert {:ok, 0} = RuvectorElixir.count(db)
      assert {:error, :not_found} = RuvectorElixir.get(db, "del_me")
    end

    test "delete nonexistent returns {:ok, false}", %{db: db} do
      assert {:ok, false} = RuvectorElixir.delete(db, "nonexistent")
    end

    test "batch insert empty list", %{db: db} do
      assert {:ok, []} = RuvectorElixir.insert_batch(db, [])
      assert {:ok, 0} = RuvectorElixir.len(db)
    end

    test "batch insert multiple entries", %{db: db} do
      entries = [
        %{id: "b1", vector: [1.0, 2.0, 3.0]},
        %{id: "b2", vector: [4.0, 5.0, 6.0]},
        %{id: "b3", vector: [7.0, 8.0, 9.0]}
      ]

      assert {:ok, ["b1", "b2", "b3"]} = RuvectorElixir.insert_batch(db, entries)
      assert {:ok, 3} = RuvectorElixir.len(db)

      assert {:ok, entry2} = RuvectorElixir.get(db, "b2")
      assert entry2.vector == [4.0, 5.0, 6.0]
    end

    test "batch insert dimension mismatch error rolls back atomically", %{db: db} do
      entries = [
        %{id: "b1", vector: [1.0, 2.0, 3.0]},
        %{id: "b2", vector: [4.0, 5.0]}
      ]

      assert {:error, reason} = RuvectorElixir.insert_batch(db, entries)
      assert String.contains?(reason, "mismatch") or String.contains?(reason, "Dimension")
      assert {:ok, 0} = RuvectorElixir.len(db)
      assert {:error, :not_found} = RuvectorElixir.get(db, "b1")
    end

    test "keys and empty? lifecycle", %{db: db} do
      assert RuvectorElixir.empty?(db) == true
      assert {:ok, []} = RuvectorElixir.keys(db)

      assert {:ok, "k1"} = RuvectorElixir.insert(db, %{id: "k1", vector: [1.0, 2.0, 3.0]})
      assert {:ok, "k2"} = RuvectorElixir.insert(db, %{id: "k2", vector: [4.0, 5.0, 6.0]})

      assert RuvectorElixir.empty?(db) == false
      assert {:ok, keys} = RuvectorElixir.keys(db)
      assert Enum.sort(keys) == ["k1", "k2"]

      assert {:ok, true} = RuvectorElixir.delete(db, "k1")
      assert {:ok, ["k2"]} = RuvectorElixir.keys(db)

      assert {:ok, true} = RuvectorElixir.delete(db, "k2")
      assert RuvectorElixir.empty?(db) == true
      assert {:ok, []} = RuvectorElixir.keys(db)
    end

    test "insert with atom ID and retrieve/delete with atom ID", %{db: db} do
      assert {:ok, "atom_key"} =
               RuvectorElixir.insert(db, %{id: :atom_key, vector: [1.0, 2.0, 3.0]})

      assert {:ok, entry} = RuvectorElixir.get(db, :atom_key)
      assert entry.id == "atom_key"
      assert entry.vector == [1.0, 2.0, 3.0]

      assert {:ok, entry_by_str} = RuvectorElixir.get(db, "atom_key")
      assert entry_by_str.id == "atom_key"

      assert {:ok, true} = RuvectorElixir.delete(db, :atom_key)
      assert {:error, :not_found} = RuvectorElixir.get(db, :atom_key)
    end

    test "insert with integer ID (including > i64::MAX) and retrieve/delete with integer ID", %{
      db: db
    } do
      assert {:ok, "42"} = RuvectorElixir.insert(db, %{id: 42, vector: [1.0, 2.0, 3.0]})

      assert {:ok, entry} = RuvectorElixir.get(db, 42)
      assert entry.id == "42"

      assert {:ok, true} = RuvectorElixir.delete(db, 42)
      assert {:error, :not_found} = RuvectorElixir.get(db, 42)

      big_id = 18_446_744_073_709_551_616
      entry_struct = VectorEntry.new([1.0, 2.0, 3.0], id: big_id)
      assert {:ok, "18446744073709551616"} = RuvectorElixir.insert(db, entry_struct)
      assert {:ok, fetched} = RuvectorElixir.get(db, big_id)
      assert fetched.id == "18446744073709551616"
      assert {:ok, true} = RuvectorElixir.delete(db, big_id)

      for boundary_id <- [-Bitwise.bsl(1, 127), Bitwise.bsl(1, 128) - 1] do
        boundary_entry = VectorEntry.new([1.0, 2.0, 3.0], id: boundary_id)
        expected_str = Integer.to_string(boundary_id)
        assert {:ok, ^expected_str} = RuvectorElixir.insert(db, boundary_entry)
        assert {:ok, found} = RuvectorElixir.get(db, boundary_id)
        assert found.id == expected_str
        assert {:ok, true} = RuvectorElixir.delete(db, boundary_id)
      end
    end

    test "get_struct and get_struct! return VectorEntry struct", %{db: db} do
      entry = VectorEntry.new([1.0, 2.0, 3.0], id: "struct_test", metadata: %{"k" => "v"})
      assert {:ok, "struct_test"} = RuvectorElixir.insert(db, entry)

      assert {:ok, %VectorEntry{} = struct} = RuvectorElixir.get_struct(db, "struct_test")
      assert struct.id == "struct_test"
      assert struct.vector == [1.0, 2.0, 3.0]
      assert struct.metadata == %{"k" => "v"}

      assert %VectorEntry{} = struct2 = RuvectorElixir.get_struct!(db, "struct_test")
      assert struct2.id == "struct_test"

      assert_raise Error, ~r/not found/i, fn ->
        RuvectorElixir.get_struct!(db, "nonexistent")
      end

      atom_meta_entry =
        VectorEntry.new([1.0, 2.0, 3.0],
          id: :doc_rt,
          metadata: %{
            status: :published,
            ok: true,
            empty: nil,
            tags: [:a, :b],
            nested: %{role: :admin}
          }
        )

      assert atom_meta_entry.metadata == %{
               "status" => "published",
               "ok" => true,
               "empty" => nil,
               "tags" => ["a", "b"],
               "nested" => %{"role" => "admin"}
             }

      assert {:ok, "doc_rt"} = RuvectorElixir.insert(db, atom_meta_entry)
      assert RuvectorElixir.get_struct!(db, :doc_rt) == atom_meta_entry
    end

    test "bang variants (insert!, insert_batch!, get!, delete!, len!, keys!, info!)", %{db: db} do
      id1 = RuvectorElixir.insert!(db, %{id: "bang1", vector: [1.0, 2.0, 3.0]})
      assert id1 == "bang1"

      ids =
        RuvectorElixir.insert_batch!(db, [
          %{id: "bang2", vector: [4.0, 5.0, 6.0]},
          %{id: "bang3", vector: [7.0, 8.0, 9.0]}
        ])

      assert ids == ["bang2", "bang3"]
      assert RuvectorElixir.len!(db) == 3
      assert RuvectorElixir.count!(db) == 3

      retrieved = RuvectorElixir.get!(db, "bang2")
      assert retrieved.id == "bang2"

      assert RuvectorElixir.delete!(db, "bang1") == true
      assert RuvectorElixir.len!(db) == 2

      keys = RuvectorElixir.keys!(db)
      assert Enum.sort(keys) == ["bang2", "bang3"]

      assert %{node_count: 2, dimensions: 3} = RuvectorElixir.info!(db)

      assert_raise Error, fn ->
        RuvectorElixir.insert!(db, %{vector: [1.0, 2.0]})
      end
    end
  end
end
