# Deep Learning, All the Way Down

Build the foundations of deep learning from first principles in modern C++.

This repository contains the code developed throughout the video series, from
tensor storage and automatic differentiation to neural networks, performance
engineering, GPU programming, and custom CUDA kernels.

[Watch the complete series on YouTube](https://www.youtube.com/playlist?list=PLZSg76FHvdTw)

## Why build this

Deep learning frameworks make sophisticated systems easy to assemble, but their
abstractions can hide the machinery underneath. This project makes that
machinery visible by implementing it one piece at a time:

- tensor shape, storage, and indexing;
- elementwise operations, reductions, and broadcasting;
- matrix multiplication and loss functions;
- computation graphs and reverse-mode automatic differentiation;
- training loops, neural network layers, and optimizers;
- CPU performance, GPU programming, and CUDA kernels.

This is an educational implementation, not a production framework or an
attempt to replace PyTorch. The goal is to understand the mathematics, data
structures, ownership decisions, and hardware behavior that mature frameworks
bring together.

## Current checkpoint

The repository currently matches the end of Episode 7.

[`main.cpp`](./main.cpp) contains the tensor implementation developed through
Episode 6. It currently supports:

- owned `double` storage with explicit shape metadata;
- rank-zero scalars and empty tensors;
- overflow-safe element counting and checked multidimensional indexing;
- mutable and const element access;
- `sum()` and `mean()` reductions;
- elementwise addition, subtraction, multiplication, and division;
- vector dot products;
- rank-two matrix multiplication;
- NumPy-style broadcasting through effective zero strides;
- mean squared error.

[`scalar_autograd.cpp`](./scalar_autograd.cpp) contains the scalar reverse-mode
automatic differentiation engine introduced in Episode 7. It adds:

- graph nodes that preserve values, operations, and parent relationships;
- lightweight `Value` handles backed by shared graph nodes;
- arithmetic operators that construct the computation graph;
- topological traversal of shared graphs;
- reverse-order gradient propagation;
- local derivative rules and gradient accumulation;
- a complete backward pass from scalar loss to weights and bias.

Scalar autograd is deliberately separate from `Tensor` for now. Connecting the
two is the next major implementation milestone.

## Episodes and code checkpoints

Every episode ends with a Git commit that preserves the exact code checkpoint
reached on screen. Earlier states remain available through the repository
history.

| Episode | What it adds | Checkpoint | Video |
| --- | --- | --- | --- |
| 1 | Tensor storage, shape, rank, invariants, and checked indexing | [`bedb564`](https://github.com/mechanical-turk/deep-learning-all-the-way-down/commit/bedb564) | [Watch](https://www.youtube.com/watch?v=DmU2b64tWfA) |
| 2 | Scalars, empty tensors, overflow checks, mutation, and `sum()` | [`b25de0c`](https://github.com/mechanical-turk/deep-learning-all-the-way-down/commit/b25de0c) | [Watch](https://www.youtube.com/watch?v=8kzL5NdxGCo) |
| 3 | Elementwise operations, dot product, and a linear prediction | [`c4d4509`](https://github.com/mechanical-turk/deep-learning-all-the-way-down/commit/c4d4509) | [Watch](https://www.youtube.com/watch?v=R_NZJ_rcX7E) |
| 4 | Rank-two matrix multiplication | [`022c750`](https://github.com/mechanical-turk/deep-learning-all-the-way-down/commit/022c750) | [Watch](https://www.youtube.com/watch?v=ZC6F284rRGo) |
| 5 | General tensor broadcasting with effective strides | [`79461fb`](https://github.com/mechanical-turk/deep-learning-all-the-way-down/commit/79461fb) | [Watch](https://www.youtube.com/watch?v=70mVVGNc0Ik) |
| 6 | Mean reduction, division, and mean squared error | [`92aa868`](https://github.com/mechanical-turk/deep-learning-all-the-way-down/commit/92aa868) | [Watch](https://www.youtube.com/watch?v=26Eg8tpM6_Q) |
| 7 | Scalar computation graphs and reverse-mode autograd | [`e692363`](https://github.com/mechanical-turk/deep-learning-all-the-way-down/commit/e692363) | [Watch](https://www.youtube.com/watch?v=QEZvHrZDSdw) |

## Build and run

The current programs are self-contained, use C++23, and depend only on the C++
standard library.

Using Clang:

```sh
clang++ -std=c++23 main.cpp -o main
./main

clang++ -std=c++23 scalar_autograd.cpp -o scalar_main
./scalar_main
```

Using GCC:

```sh
g++ -std=c++23 main.cpp -o main
./main

g++ -std=c++23 scalar_autograd.cpp -o scalar_main
./scalar_main
```

Both programs verify their behavior with assertions. A successful run ends
with:

```text
Success!
```

## Explore an earlier episode

Check out any episode's commit to inspect or run the implementation exactly as
it existed at that checkpoint:

```sh
git switch --detach bedb564
clang++ -std=c++23 main.cpp -o main
./main
```

Return to the latest checkpoint with:

```sh
git switch main
```

## Repository layout

```text
.
├── main.cpp             # cumulative Tensor implementation
├── scalar_autograd.cpp  # scalar reverse-mode autograd
├── .clang-format        # formatting rules used in the series
├── .clangd              # clangd configuration
└── .vscode              # editor settings used while recording
```

The code remains intentionally compact while the foundations are still being
established. The structure will evolve when the implementation reaches the
point where separating interfaces, implementations, tests, and benchmarks
improves understanding rather than hiding it.

## Direction

The next stages will connect automatic differentiation to tensors and use it to
train small models. From there, the project will build toward neural network
layers, optimizers, attention, transformers, systems profiling, GPU execution,
and custom CUDA kernels.

The implementation will be redesigned whenever a new capability exposes a
weakness in the current representation. Those redesigns are part of the
material.

## Feedback

Technical feedback is welcome, especially around correctness, C++ API design,
ownership, numerical behavior, and performance. Open a GitHub issue with a
small example when possible.
