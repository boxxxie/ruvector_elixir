defmodule RuvectorElixir.DistanceTest do
  use ExUnit.Case, async: true

  describe "distance calculations (unit tests matching upstream distance_tests)" do
    test "euclidean distance: distance to self is ~0" do
      v = [1.0, 2.0, 3.0]
      assert {:ok, dist} = RuvectorElixir.distance(v, v, :euclidean)
      assert_in_delta dist, 0.0, 0.001
    end

    test "euclidean distance: orthogonal vectors sqrt(2) ~ 1.414" do
      a = [1.0, 0.0, 0.0]
      b = [0.0, 1.0, 0.0]
      assert {:ok, dist} = RuvectorElixir.distance(a, b, :euclidean)
      assert_in_delta dist, 1.4142, 0.01
    end

    test "cosine distance: parallel vectors have ~0 distance" do
      a = [1.0, 2.0, 3.0]
      b = [2.0, 4.0, 6.0]
      assert {:ok, dist} = RuvectorElixir.distance(a, b, :cosine)
      assert_in_delta dist, 0.0, 0.001
    end

    test "cosine distance: orthogonal vectors have distance ~1" do
      a = [1.0, 0.0, 0.0]
      b = [0.0, 1.0, 0.0]
      assert {:ok, dist} = RuvectorElixir.distance(a, b, :cosine)
      assert_in_delta dist, 1.0, 0.001
    end

    test "dot product distance: negative distance for similar vectors" do
      a = [1.0, 2.0, 3.0]
      b = [1.0, 2.0, 3.0]
      assert {:ok, dist} = RuvectorElixir.distance(a, b, :dot_product)
      assert dist < 0.0
    end

    test "manhattan distance: e.g. [0,0,0] and [1,1,1] is 3.0" do
      a = [0.0, 0.0, 0.0]
      b = [1.0, 1.0, 1.0]
      assert {:ok, dist} = RuvectorElixir.distance(a, b, :manhattan)
      assert_in_delta dist, 3.0, 0.001
    end

    test "dimension mismatch returns error" do
      a = [1.0, 2.0]
      b = [1.0, 2.0, 3.0]
      assert {:error, reason} = RuvectorElixir.distance(a, b, :euclidean)
      assert String.contains?(reason, "mismatch") or String.contains?(reason, "Dimension")
    end

    test "convenience distance/2 helper defaults to cosine" do
      a = [1.0, 0.0]
      b = [0.0, 1.0]
      assert {:ok, dist} = RuvectorElixir.distance(a, b)
      assert_in_delta dist, 1.0, 0.001
    end

    test "accepts string metric names" do
      a = [0.0, 0.0]
      b = [3.0, 4.0]
      assert {:ok, dist} = RuvectorElixir.distance(a, b, "euclidean")
      assert_in_delta dist, 5.0, 0.001
    end

    test "accepts integer vectors and mixed types" do
      a = [1, 2, 3]
      b = [1.0, 2.0, 3.0]
      assert {:ok, dist} = RuvectorElixir.distance(a, b, :euclidean)
      assert_in_delta dist, 0.0, 0.001
    end

    test "rejects unrecognized metric and non-finite vectors" do
      assert_raise ArgumentError, ~r/Unsupported distance metric/, fn ->
        RuvectorElixir.distance([1.0, 0.0], [0.0, 1.0], :l2)
      end

      assert {:error, "Invalid vector a"} =
               RuvectorElixir.distance([1.0e100, 0.0], [0.0, 1.0], :euclidean)
    end

    test "distance! returns float or raises RuvectorElixir.Error" do
      a = [0.0, 0.0]
      b = [3.0, 4.0]
      assert RuvectorElixir.distance!(a, b, :euclidean) == 5.0

      assert_raise RuvectorElixir.Error, fn ->
        RuvectorElixir.distance!([1.0], [1.0, 2.0], :euclidean)
      end
    end
  end
end
