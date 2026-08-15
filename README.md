# Deep Learning, All the Way Down

A code-first journey through deep learning, from tensor storage to custom GPU kernels, built from first principles in modern C++.

This repository contains the evolving implementation developed throughout the
video series. Each episode extends the same small tensor rather than replacing
it with a separate finished library:

[Watch Deep Learning, All the Way Down on YouTube](https://www.youtube.com/playlist?list=PLZSg76FHvdTw)

## Why this series exists

Deep learning libraries make sophisticated models remarkably easy to build. That convenience can also hide the machinery underneath:

- How does a tensor represent multidimensional data?
- How are operations evaluated across shapes?
- How does automatic differentiation construct and traverse a computation graph?
- How do gradients become parameter updates?
- How are neural networks assembled from these foundations?
- What changes when computation moves from the CPU to the GPU?
- How are custom CUDA kernels designed, measured, and optimized?

This series answers those questions by building the machinery ourselves.

By the end, we will end up with something that resembles a PyTorch clone. The goal is to develop a rigorous understanding of the mathematical, numerical, and systems concepts that libraries like PyTorch bring together.

## The journey

We will work progressively through:

1. Tensor representation and indexing
2. Numeric types and generic tensor storage
3. Tensor operations and broadcasting
4. Views, strides, and memory layouts
5. Automatic differentiation
6. Optimization and training loops
7. Neural network layers
8. Attention and transformers
9. Language models
10. GPU programming with CUDA
11. Custom GPU kernels
12. Profiling and performance optimization

Each chapter builds on foundations established by the chapters before it.

## Episodes

### Episode 1: Building a Tensor from Scratch

The first chapter begins with a deliberately small tensor implementation in a single C++ file.

It introduces:

- Flat tensor storage
- Shapes and dimensions
- Rank
- Number of elements
- Representation invariants
- Checked multidimensional indexing
- Row-major index calculation
- Basic validation using assertions and exceptions

The first version intentionally supported only:

- `double` values
- Tensors with at least one dimension
- Positive dimension sizes
- Owned, contiguous, row-major storage
- Read-only element access

Features such as scalar tensors, empty tensors, mutation, reductions, tensor
operations, views, and autograd were deferred so the foundational representation
could remain visible.

[Watch Episode 1](https://www.youtube.com/watch?v=DmU2b64tWfA)

### Episode 2: Completing Our Tensor in C++

Episode 2 expands that initial representation with:

- Rank-zero scalar tensors
- Empty tensors and zero-length dimensions
- Overflow-safe element counting and indexing
- Checked dimension queries
- Mutable and const element access
- Full-tensor summation as the first reduction

The current [`main.cpp`](./main.cpp) contains the implementation as it exists
after Episode 2, together with its assertion-based verification. Earlier episode
states remain available through the repository's Git history.

## Build and run the current implementation

The implementation remains self-contained in one file and requires only a C++23
compiler.

Using Clang:

```bash
clang++ -std=c++23 main.cpp -o main
./main
```

Using GCC:

```bash
g++ -std=c++23 main.cpp -o main
./main
```

A successful run ends with:

```text
Success!
```

No build system or external dependencies are required.

## Design philosophy

This project favors understanding over convenience.

We will begin with small, transparent implementations before introducing more powerful abstractions. As the code becomes more capable, we will examine why each new abstraction is needed, what problem it solves, and what tradeoffs it introduces.

The project will emphasize:

- Mathematical derivation
- Explicit invariants
- Modern C++ design
- Correctness testing
- Numerical behavior
- Memory representation
- Performance measurement
- CPU and GPU architecture

Some early implementations will eventually be redesigned or replaced. That progression is part of the material.

## Follow the series

Watch the complete playlist:

[Deep Learning, All the Way Down](https://www.youtube.com/playlist?list=PLZSg76FHvdTw)

New chapters will continue expanding the implementation from a rudimentary tensor into a working deep learning system.
