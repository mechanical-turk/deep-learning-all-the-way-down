// chapter 1: Rudimentary tensor


// stores double values
// stores one or more positive dimensions
// owns one flat vector
// reports shape, rank, and number of elements
// rejects mismatched shape and data
// reads values using checked multidimensional coordinates

// Later:
// scalars
// empty tensors
// mutations
// operations (+, -, *)
// generic types
// autograd

#include <cstddef>
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
    if (shape_.empty()) {
      throw std::invalid_argument("chapter 1 tensors must have at least one dimension");
    }

    std::size_t expected_elements = 1;

    for (const std::size_t dimension: shape_) {
      if (dimension == 0) {
        throw std::invalid_argument("chapter 1 tensor dimensions must be positive");
      }
      expected_elements *= dimension;
      // TODO: handle overflow
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

  [[nodiscard]] double at(const std::vector<std::size_t>& idx) const {
    if (idx.size() != rank()) {
      throw std::invalid_argument("number of indices must match tensor rank");
    }

    std::size_t flat_index = 0;
    std::size_t stride = 1;

    for (std::size_t axis = rank(); axis > 0; --axis) {
      const std::size_t current_axis = axis - 1;

      if (idx[current_axis] >= shape_[current_axis]) {
        throw std::out_of_range("tensor index is outside its dimension");
      }

      flat_index += idx[current_axis] * stride;
      stride *= shape_[current_axis];
      // TODO: handle overflow
    }

    return data_[flat_index];
  }

private:
  std::vector<std::size_t> shape_;
  std::vector<double> data_;

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
    assert(std::string(error.what()) == "chapter 1 tensors must have at least one dimension");
    test_5_threw = true;
  }
  assert(test_5_threw);

  std::cout << "Test 6\n";
  bool test_6_threw = false;
  try {
    Tensor({1, 0}, {});
  } catch (std::invalid_argument& error) {
    assert(std::string(error.what()) == "chapter 1 tensor dimensions must be positive");
    test_6_threw = true;
  }
  assert(test_6_threw);

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


  std::cout << "Success!\n";
  return 0;
}