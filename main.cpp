#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>
#include <cassert>


class Tensor {
public:
  Tensor(
    std::vector<std::size_t> shape,
    std::vector<double> data
  ) : shape_(std::move(shape)), data_(std::move(data)) {
    std::size_t expected_elements;

    bool has_zero_dims = false;

    for (const std::size_t dimension: shape_) {
      if (dimension == 0) {
        has_zero_dims = true;
        expected_elements = 0;
        break;
      }
    }

    if (!has_zero_dims) {
      expected_elements = 1;
      for (const std::size_t dimension: shape_) {
        if (expected_elements > std::numeric_limits<std::size_t>::max() / dimension) {
          throw std::overflow_error("tensor element count overflows size_t");
        }

        expected_elements *= dimension;
      }
    }

    if (expected_elements != data_.size()) {
      throw std::invalid_argument("tensor shape does not match its data");
    }
  }
  
  [[nodiscard]] const std::vector<std::size_t>& shape() const noexcept {
    return shape_;
  }

  [[nodiscard]] const std::vector<double>& data() const noexcept {
    return data_;
  }

  [[nodiscard]] std::size_t rank() const noexcept {
    return shape_.size();
  }

  [[nodiscard]] std::size_t numel() const noexcept {
    return data_.size();
  }

  [[nodiscard]] std::size_t dimension(const std::size_t axis) const {
    if (axis >= rank()) {
      throw std::out_of_range("tensor axis is outside its rank");
    }
    return shape_[axis];
  }

  double& at(const std::vector<std::size_t>& idx) {
    return data_[flat_index(idx)];
  }

  [[nodiscard]] double at(const std::vector<std::size_t>& idx) const {
    return data_[flat_index(idx)];
  }

  [[nodiscard]] Tensor sum() const {
    double result = 0.0;
    for (const double value: data_) {
      result += value;
    }

    return Tensor({}, {result});
  }

  [[nodiscard]] Tensor dot(const Tensor& other) const {
    if (rank() != 1 || other.rank() != 1) {
      throw std::invalid_argument("dot requires two rank-one tensors");
    }

    if (shape_ != other.shape_) {
      throw std::invalid_argument("dot requires vectors of equal length");
    }

    // TODO: (performance): Benchmark a fused dot-product kernel that avoids
    // allocating and traversing an intermediate product tensor.

    return (*this * other).sum();
  }

  [[nodiscard]] Tensor operator+(const Tensor& other) const {
    return elementwise_binary(
      other,
      [](const double left, const double right) {
        return left + right;
      }
    );
  }

  [[nodiscard]] Tensor operator-(const Tensor& other) const {
    return elementwise_binary(
      other,
      [](const double left, const double right) {
        return left - right;
      }
    );
  }

  [[nodiscard]] Tensor operator*(const Tensor& other) const {
    return elementwise_binary(
      other,
      [](const double left, const double right) {
        return left * right;
      }
    );
  }

  [[nodiscard]] Tensor matmul(const Tensor& other) const {
    if (rank() != 2 || other.rank() != 2) {
      throw std::invalid_argument("matmul requires two rank-two tensors");
    }

    const std::size_t left_rows = shape_[0];
    const std::size_t left_cols = shape_[1];
    const std::size_t right_rows = other.shape_[0];
    const std::size_t right_cols = other.shape_[1];

    if (left_cols != right_rows) {
      throw std::invalid_argument("matmul inner dimensions must match");
    }

    const std::vector<std::size_t> result_shape{
      left_rows,
      right_cols
    };

    std::vector<double> result_data(
      left_rows * right_cols,
      0.0
    );

    for (std::size_t row = 0; row < left_rows; ++row) {
      for (std::size_t col = 0; col < right_cols; ++col) {
        double sum = 0.0;
        for (std::size_t index = 0; index < left_cols; ++index) {
          sum += 
            data_[row * left_cols + index] * 
            other.data_[index * right_cols + col];
        }
        result_data[row * right_cols + col] = sum;
      }
    }

    return Tensor(
      std::move(result_shape),
      std::move(result_data)
    );

  }


private:
  std::vector<std::size_t> shape_;
  std::vector<double> data_;


  [[nodiscard]] Tensor elementwise_binary(
    const Tensor& other,
    const auto& operation
  ) const {

    std::vector<std::size_t> result_shape = broadcast_shape(
      shape_, other.shape_
    );

    const std::vector<std::size_t> left_strides = effective_strides(
      shape_, strides(), result_shape.size()
    );

    const std::vector<std::size_t> right_strides = effective_strides(
      other.shape_, other.strides(), result_shape.size()
    );

    std::size_t result_numel = 1;
    for (const std::size_t dim: result_shape) {
      result_numel *= dim;
    }

    std::vector<double> result_data(result_numel);
    std::vector<std::size_t> coordinate(result_shape.size(), 0);

    for (std::size_t flat = 0; flat < result_numel; ++flat) {
      const std::size_t left_index = stride_offset(coordinate, left_strides);
      const std::size_t right_index = stride_offset(coordinate, right_strides);

      result_data[flat] = operation(data_[left_index], other.data_[right_index]);
      advance_coordinate(coordinate, result_shape);
    }

    return Tensor(std::move(result_shape), std::move(result_data));
  }

  [[nodiscard]] static std::size_t stride_offset(
    const std::vector<std::size_t>& coordinate,
    const std::vector<std::size_t>& effective_strides
  ) {
    std::size_t index = 0;
    for (std::size_t axis = 0; axis < coordinate.size(); ++axis) {
      index += coordinate[axis] * effective_strides[axis];
    }
    return index;
  }

  static void advance_coordinate(
    std::vector<std::size_t>& coordinate,
    const std::vector<std::size_t>& shape
  ) {
    for (std::size_t axis = coordinate.size(); axis-- > 0;) {
      if (++coordinate[axis] < shape[axis]) {
        return;
      }
      coordinate[axis] = 0;
    }
  }

  [[nodiscard]] std::vector<std::size_t> strides() const {
    std::vector<std::size_t> result(shape_.size());
    std::size_t stride = 1;

    for (std::size_t axis = rank(); axis-- > 0;) {
      result[axis] = stride;
      stride *= shape_[axis];
    }

    return result;
  }

  [[nodiscard]] static std::vector<std::size_t> broadcast_shape(
    const std::vector<std::size_t>& left,
    const std::vector<std::size_t>& right
  ) {

    const std::size_t rank = std::max(left.size(), right.size());
    std::vector<std::size_t> result(rank);

    const std::size_t left_offset = rank - left.size();
    const std::size_t right_offset = rank - right.size();

    for (std::size_t axis = 0; axis < rank; ++axis) {
      const bool left_has_axis = axis >= left_offset;
      const bool right_has_axis = axis >= right_offset;

      std::size_t left_dim = 1;
      if (left_has_axis) {
        const std::size_t left_axis = axis - left_offset;
        left_dim = left[left_axis];
      }

      std::size_t right_dim = 1;
      if (right_has_axis) {
        const std::size_t right_axis = axis - right_offset;
        right_dim = right[right_axis];
      }

      const bool sizes_match = left_dim == right_dim;
      const bool left_can_stretch = left_dim == 1;
      const bool right_can_stretch = right_dim == 1;

      if (!sizes_match && !left_can_stretch && !right_can_stretch) {
        throw std::invalid_argument(
          "cannot broadcast - shape mismatch at dimension " + std::to_string(axis)
        );
      }
      result[axis] = std::max(left_dim, right_dim);
    }

    return result;
  }

  [[nodiscard]] static std::vector<std::size_t> effective_strides(
    const std::vector<std::size_t>& shape,
    const std::vector<std::size_t>& own_strides,
    const std::size_t target_rank
  ) {
    std::vector<std::size_t> result(target_rank, 0);
    const std::size_t offset = target_rank - shape.size();

    for (std::size_t axis = 0; axis < shape.size(); ++axis) {
      if (shape[axis] != 1) {
        result[offset + axis] = own_strides[axis];
      }
    }

    return result;
    
  }

  [[nodiscard]] std::size_t flat_index(const std::vector<std::size_t>& idx) const {
    if (idx.size() != rank()) {
      throw std::invalid_argument("number of indices must match tensor rank");
    }

    for (std::size_t axis = 0; axis < rank(); ++axis) {
      if (idx[axis] >= shape_[axis]) {
        throw std::out_of_range("tensor index is outside its dimension");
      }
    }

    std::size_t flat_index = 0;
    std::size_t stride = 1;

    for (std::size_t axis = rank(); axis > 0; --axis) {
      const std::size_t current_axis = axis - 1;
      flat_index += idx[current_axis] * stride;
      stride *= shape_[current_axis];
    }

    return flat_index;
  }

};

int main() {

  std::vector<std::size_t> shape_1 {2, 3};
  std::vector<double> data_1 {0,1,2,3,4,5};

  Tensor t(
    shape_1,
    data_1
  );

  std::cout << "Test 1\n";
  auto& actual_shape_1 = t.shape();
  assert(actual_shape_1 == shape_1);


  std::cout << "Test 2\n";
  auto& actual_data_1 = t.data();
  assert(actual_data_1 == data_1);

 
  std::cout << "Test 3\n";
  std::size_t actual_rank_1 = t.rank();
  assert(actual_rank_1 == 2);

  std::cout << "Test 4\n";
  std::size_t actual_numel_1 = t.numel();
  assert(actual_numel_1 == 6);

  std::cout << "Test 5\n";
  bool test_5_threw = false;
  try {
    Tensor({}, {});
  } catch (std::invalid_argument& error) {
    assert(std::string(error.what()) == "tensor shape does not match its data");
    test_5_threw = true;
  }
  assert(test_5_threw);
  Tensor scalar({}, {5});
  assert(scalar.rank() == 0);
  assert(scalar.numel() == 1);
  assert(scalar.shape() == std::vector<std::size_t>{});

  std::cout << "Test 6\n";
  Tensor empty_matrix({1, 0}, {});
  assert(empty_matrix.rank() == 2);
  assert(empty_matrix.numel() == 0);
  assert((empty_matrix.shape() == std::vector<std::size_t>{1, 0}));

  std::cout << "Test 7\n";
  bool test_7_threw = false;
  try {
    Tensor({2, 3}, {1,2,3});
  } catch (std::invalid_argument& error) {
    assert(std::string(error.what()) == "tensor shape does not match its data");
    test_7_threw = true;
  }
  assert(test_7_threw);

  std::cout << "Test 8\n";
  double val = t.at({1, 1});
  assert(val == 4);

  std::cout << "Test 9\n";
  bool test_9_threw = false;
  try {
    double val_9 = t.at({1,2,3});
  } catch (std::invalid_argument& error) {
    assert(std::string(error.what()) == "number of indices must match tensor rank");
    test_9_threw = true;
  }
  assert(test_9_threw);

  std::cout << "Test 10\n";
  bool test_10_threw = false;
  try {
    double val_10 = t.at({2,3});
  } catch (std::out_of_range& error) {
    assert(std::string(error.what()) == "tensor index is outside its dimension");
    test_10_threw = true;
  }
  assert(test_10_threw);

  std::cout << "Test 11\n";
  bool test_11_threw = false;
  try {
    Tensor overflow({std::numeric_limits<std::size_t>::max(), 2}, {});
  } catch (std::overflow_error& error) {
    assert(std::string(error.what()) == "tensor element count overflows size_t");
    test_11_threw = true;
  }
  assert(test_11_threw);

  std::cout << "Test 12\n";
  Tensor zero_and_max({std::numeric_limits<std::size_t>::max(), 0}, {});
  assert(zero_and_max.rank() == 2);
  assert(zero_and_max.numel() == 0);

  std::cout << "Test 13\n";
  Tensor zero_and_max_3d({std::numeric_limits<std::size_t>::max(), 2, 0}, {});
  assert(zero_and_max_3d.rank() == 3);
  assert(zero_and_max_3d.numel() == 0);

  std::cout << "Test 14\n";
  Tensor dim_test_tensor({2, 0, 3}, {});
  assert(dim_test_tensor.dimension(0) == 2);
  assert(dim_test_tensor.dimension(1) == 0);
  assert(dim_test_tensor.dimension(2) == 3);

  bool rejected_axis = false;
  try {
    auto d = dim_test_tensor.dimension(3);
  } catch(const std::out_of_range&) {
    rejected_axis = true;
  }
  assert(rejected_axis);

  std::cout << "Test 15\n";
  Tensor mutation_tensor({2,3}, {0,1,2,3,4,5});
  assert(mutation_tensor.at({0, 0}) == 0);
  mutation_tensor.at({0, 0}) = 5;
  assert(mutation_tensor.at({0, 0}) == 5);

  std::cout << "Test 16\n";
  Tensor scalar_16({}, {10});
  assert(10 == scalar_16.at({}));

  std::cout << "Test 17\n";
  assert((Tensor({2, 3}, {1,2,3,4,5,6}).sum().at({}) == 21.0));
  assert((Tensor({}, {7.0}).sum().at({}) == 7.0));
  assert((Tensor({1}, {7.0}).sum().at({}) == 7.0));
  assert((Tensor({0}, {}).sum().at({}) == 0.0));
  assert((Tensor({2, 0, 3}, {}).sum().at({}) == 0.0));

  std::cout << "Test 18\n";
  const Tensor left({3}, {1.0, 2.0, 3.0});
  const Tensor right({3}, {10.0, 20.0, 30.0});

  const Tensor added = left + right;
  assert((added.shape() == std::vector<std::size_t>{3}));
  assert((added.data() == std::vector<double>{11.0, 22.0, 33.0}));
  assert((left.data() == std::vector<double>{1.0, 2.0, 3.0}));
  assert((right.data() == std::vector<double>{10.0, 20.0, 30.0}));

  const Tensor scalar_sum = Tensor({}, {2.0}) + Tensor({}, {3.0});
  assert(scalar_sum.rank() == 0);
  assert(scalar_sum.at({}) == 5.0);

  const Tensor empty_sum = Tensor({0}, {}) + Tensor({0}, {});
  assert((empty_sum.shape() == std::vector<std::size_t>{0}));
  assert(empty_sum.numel() == 0);

  std::cout << "Test 19\n";

  bool rejected_mismatched_shapes = false;

  try {
    const auto invalid = Tensor({3}, {1.0, 2.0, 3.0}) + Tensor({1, 2}, {3.0, 4.0});
  } catch (const std::invalid_argument&) {
    rejected_mismatched_shapes = true;
  }
  assert(rejected_mismatched_shapes);

  std::cout << "Test 20\n";

  const Tensor arithmetic_left({3}, {2.0, 3.0, 4.0});
  const Tensor arithmetic_right({3}, {5.0, 6.0, 7.0});

  const Tensor arithmetic_sum = arithmetic_left + arithmetic_right;
  const Tensor arithmetic_sub = arithmetic_left - arithmetic_right;
  const Tensor arithmetic_prod = arithmetic_left * arithmetic_right;

  assert((arithmetic_sum.data() == std::vector<double>{7.0, 9.0, 11.0}));
  assert((arithmetic_sub.data() == std::vector<double>{-3.0, -3.0, -3.0}));
  assert((arithmetic_prod.data() == std::vector<double>{10.0, 18.0, 28.0}));

  std::cout << "Test 21\n";

  const Tensor dot_result = Tensor({3}, {2.0, 3.0, 4.0}).dot(Tensor({3}, {5.0, 6.0, 7.0}));

  assert(dot_result.rank() == 0);
  assert(dot_result.at({}) == 56.0);

  const Tensor empty_dot = Tensor({0}, {}).dot(Tensor({0}, {}));
  assert(empty_dot.rank() == 0);
  assert(empty_dot.at({}) == 0);

  bool rejected_matrix_dot = false;

  try {
    const auto invalid = Tensor({1, 2}, {1.0, 2.0}).dot(Tensor({1, 2}, {3.0, 4.0}));
  } catch (const std::invalid_argument&) {
    rejected_matrix_dot = true;
  }
  assert(rejected_matrix_dot);

  bool rejected_unequal_lengths = false;

  try {
    const auto invalid = Tensor({2}, {1.0, 2.0}).dot(Tensor({3}, {3.0, 4.0, 5.0}));
  } catch (const std::invalid_argument&) {
    rejected_unequal_lengths = true;
  }
  assert(rejected_unequal_lengths);


  // features = [ 4.0, 3.0, 2.0]
  // weights  = [0.5,  -1.0, 2.0]
  // scaled   = [2.0,  -3.0, 4.0]
  // weighted_sum = 3.0
  // bias = 0.5
  // prediction = 3.0 + 0.5 = 3.5
  // pred = weights.dot(features) + bias

  std::cout << "Test 22\n";

  const Tensor features({3}, {4.0, 3.0, 2.0});

  const Tensor weights_({3}, {0.5, -1.0, 2.0});
  const Tensor bias({}, {0.5});

  const Tensor prediction = weights_.dot(features) + bias;
  assert(prediction.rank() == 0);
  assert(prediction.at({}) == 3.5);

  


  
  // 2 observations
  // 3 features 

  // 2x3 matrix

  const Tensor inputs(
    {2, 3},
    {
      4.0, 3.0, 2.0,
      1.0, 2.0, 0.5
    }
  );

  const Tensor weights(
    {3, 1},
    {
      0.5,
      -1.0,
      2.0
    }
  );

  // 3.0
  // -0.5

  std::cout << "Test 23\n";
  const Tensor matmul_23 = inputs.matmul(weights);

  assert((
    matmul_23.shape() == std::vector<std::size_t>{2, 1}
  ));

  assert((
    matmul_23.data() == std::vector<double>{3.0, -0.5}
  ));

  const Tensor pred_23 = matmul_23 + bias;
  assert((
    pred_23.data() == std::vector<double>{3.5, 0.0}
  ));





  std::cout << "Success!\n";
  return 0;
}