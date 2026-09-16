defmodule RuvectorElixir.StorageTest do
  use ExUnit.Case, async: true

  alias RuvectorElixir.VectorEntry

  setup do
    db_path =
      Path.join(System.tmp_dir!(), "ruvector_storage_#{System.unique_integer([:positive])}.rvf")

    on_exit(fn ->
      File.rm_rf(db_path)
    end)

    {:ok, db} = RuvectorElixir.open(db_path, 3)
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
      assert {:ok, retrieved} = RuvectorElixir.get_entry(db, "struct_entry")
      assert retrieved.id == "struct_entry"
      assert retrieved.vector == [1.0, 2.0, 3.0]
      assert retrieved.metadata == %{"tag" => "v1"}
    end

    test "insert auto generates id when omitted or nil", %{db: db} do
      # Test map with nil id
      assert {:ok, id1} = RuvectorElixir.insert(db, %{vector: [1.0, 2.0, 3.0]})
      assert is_binary(id1)
      assert String.contains?(id1, "-")

      # Test raw vector list
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
      {:ok, "del_me"} = RuvectorElixir.insert(db, %{id: "del_me", vector: [1.0, 2.0, 3.0]})
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

    test "batch insert dimension mismatch error", %{db: db} do
      entries = [
        %{id: "b1", vector: [1.0, 2.0, 3.0]},
        %{id: "b2", vector: [4.0, 5.0]}
      ]

      assert {:error, reason} = RuvectorElixir.insert_batch(db, entries)
      assert String.contains?(reason, "mismatch") or String.contains?(reason, "Dimension")
    end

    test "all_ids and empty? lifecycle", %{db: db} do
      assert {:ok, true} = RuvectorElixir.empty?(db)
      assert {:ok, []} = RuvectorElixir.keys(db)

      RuvectorElixir.insert(db, %{id: "k1", vector: [1.0, 2.0, 3.0]})
      RuvectorElixir.insert(db, %{id: "k2", vector: [4.0, 5.0, 6.0]})

      assert {:ok, false} = RuvectorElixir.empty?(db)
      assert {:ok, keys} = RuvectorElixir.keys(db)
      assert Enum.sort(keys) == ["k1", "k2"]

      RuvectorElixir.delete(db, "k1")
      assert {:ok, ["k2"]} = RuvectorElixir.keys(db)

      RuvectorElixir.delete(db, "k2")
      assert {:ok, true} = RuvectorElixir.empty?(db)
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

    test "insert with integer ID and retrieve/delete with integer ID", %{db: db} do
      assert {:ok, "42"} = RuvectorElixir.insert(db, %{id: 42, vector: [1.0, 2.0, 3.0]})

      assert {:ok, entry} = RuvectorElixir.get(db, 42)
      assert entry.id == "42"

      assert {:ok, true} = RuvectorElixir.delete(db, 42)
      assert {:error, :not_found} = RuvectorElixir.get(db, 42)
    end

    test "get_struct and get_struct! return VectorEntry struct", %{db: db} do
      entry = VectorEntry.new([1.0, 2.0, 3.0], id: "struct_test", metadata: %{"k" => "v"})
      {:ok, "struct_test"} = RuvectorElixir.insert(db, entry)

      assert {:ok, %VectorEntry{} = struct} = RuvectorElixir.get_struct(db, "struct_test")
      assert struct.id == "struct_test"
      assert struct.vector == [1.0, 2.0, 3.0]
      assert struct.metadata == %{"k" => "v"}

      assert %VectorEntry{} = struct2 = RuvectorElixir.get_struct!(db, "struct_test")
      assert struct2.id == "struct_test"

      assert_raise RuntimeError, ~r/not found/, fn ->
        RuvectorElixir.get_struct!(db, "nonexistent")
      end
    end

    test "bang variants (insert!, insert_batch!, get!, delete!, len!, keys!)", %{db: db} do
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

      assert_raise ArgumentError, fn ->
        RuvectorElixir.insert!(db, %{vector: [1.0, 2.0]})
      end
    end
  end
end
