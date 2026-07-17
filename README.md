# chora
A minimal tensor library in Rust. Personal learning project.

## Structure
```
src/
  backend/
    mod.rs   Backend trait, associated Storage type
    cpu.rs   CPU backend
    gpu.rs   GPU backend (planned)
  ops/       tensor operations
    mod.rs 
    elementwise.rs   elementwise ops
    matmul.rs        matrix multiplication
    reduce.rs        reduction ops
    unary.rs         unary ops
  tensor.rs  Tensor struct
  error.rs   error types
  lib.rs
```

## Journal
July 15, 2026
- Created project structure and started working on backend, tensor struct
- Need to learn how to make and implement error types

July 16, 2026
- Added constructors and methods from old tensor project and redesigned them
- implemented error types
- decided to hold off on Display trait for Tensors
- started ops

July 17, 2026
- Created storage(), from_storage(), map(), binary_map() to assist ops
  - storage() returns the storage of a tensor
  - from_storage() creates a tensor from existing shape and data
  - map() maps a function to the elements of a tensor
  - binary_map() takes two tensors, zips them together, applies some function to the two elements, and then collects the elements and wraps them in a new tensor
- created elementwise and reduction ops
