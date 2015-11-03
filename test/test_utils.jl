facts("Utilities") do

  T = _kruskal3_generator(2, (10, 20, 30), 1, true)
  @fact size(T) --> (10, 20, 30)

  context("_row_unfold()") do
    res = TensorDecompositions._row_unfold(T, 1)
    @fact size(res) --> (10, 600)

    res = TensorDecompositions._row_unfold(T, 2)
    @fact size(res) --> (20, 300)

    res = TensorDecompositions._row_unfold(T, 3)
    @fact size(res) --> (30, 200)
  end

  context("_col_unfold()") do
    res = TensorDecompositions._col_unfold(T, 1)
    @fact size(res) --> (600, 10)

    res = TensorDecompositions._col_unfold(T, 2)
    @fact size(res) --> (300, 20)

    res = TensorDecompositions._col_unfold(T, 3)
    @fact size(res) --> (200, 30)
  end

end
