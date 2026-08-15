#include <cstddef>
#include <limits>
#include <stdexcept>
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

  [[nodiscard]] double sum() const noexcept {
    // ^ TODO: return tensor

    double result = 0.0;
    for (const double value: data_) {
      result += value;
    }

    return result;
  }

private:
  std::vector<std::size_t> shape_;
  std::vector<double> data_;

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
  assert((Tensor({2, 3}, {1,2,3,4,5,6}).sum() == 21.0));
  assert((Tensor({}, {7.0}).sum() == 7.0));
  assert((Tensor({1}, {7.0}).sum() == 7.0));
  assert((Tensor({0}, {}).sum() == 0.0));
  assert((Tensor({2, 0, 3}, {}).sum() == 0.0));

  std::cout << "Success!\n";
  return 0;
}