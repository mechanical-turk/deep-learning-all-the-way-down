#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct MnistBatch {
  std::vector<double> pixels;
  std::vector<std::size_t> labels;
};

class MnistReader {
public:
  MnistReader(const std::string& images, const std::string& labels)
    : images_(images, std::ios::binary), labels_(labels, std::ios::binary) {
    if (!images_) throw std::runtime_error("Cannot open " + images + ". Run bash scripts/download-mnist.sh, then ./main from the repo root.");
    if (!labels_) throw std::runtime_error("Cannot open " + labels + ". Run bash scripts/download-mnist.sh, then ./main from the repo root.");
    const auto image_format = read_u32(images_);
    remaining_ = read_u32(images_);
    const auto rows = read_u32(images_);
    const auto columns = read_u32(images_);
    const auto label_format = read_u32(labels_);
    const auto label_count = read_u32(labels_);
    if (image_format != 2051 || label_format != 2049 ||
        rows != 28 || columns != 28 || label_count != remaining_)
      throw std::runtime_error("invalid MNIST headers");
  }

  [[nodiscard]] MnistBatch read_batch(std::size_t batch_size) {
    if (batch_size == 0)
      throw std::invalid_argument("batch size must be positive");
    const std::size_t count = std::min(batch_size, remaining_);
    std::vector<unsigned char> pixels(count * 784);
    std::vector<unsigned char> digits(count);
    if (count > 0 &&
        (!images_.read(reinterpret_cast<char*>(pixels.data()), pixels.size()) ||
         !labels_.read(reinterpret_cast<char*>(digits.data()), digits.size())))
      throw std::runtime_error("incomplete MNIST records");

    std::vector<double> values(pixels.size());
    for (std::size_t index = 0; index < pixels.size(); ++index)
      values[index] = pixels[index] / 255.0;
    std::vector<std::size_t> labels(digits.begin(), digits.end());
    for (std::size_t label : labels)
      if (label > 9) throw std::runtime_error("invalid digit label");

    remaining_ -= count;
    return {std::move(values), std::move(labels)};
  }

private:
  [[nodiscard]] static std::uint32_t read_u32(std::istream& file) {
    unsigned char bytes[4];
    if (!file.read(reinterpret_cast<char*>(bytes), 4))
      throw std::runtime_error("incomplete MNIST header");
    std::uint32_t value = 0;
    for (unsigned char byte : bytes)
      value = value * 256 + byte;
    return value;
  }

  std::ifstream images_;
  std::ifstream labels_;
  std::size_t remaining_;
};
