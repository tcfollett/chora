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

July 18, 2026

TODOS
- [x] from_storage() needs to validate data.len() == shape element count
- [x] fix ops which are closure based
  - need per op Backend methods because closures cant run on GPU
  1. replace all ops that have closures
  2. remove map() and binary_map() and add one method for each op in Backend
  3. update CpuBackend with same logic as map but with fixed functions
  4. change ops to call methods
- [x] unary edge cases like sqrt of negative (going to let this go to NaN)
- [x] division by zero handling (going to let this go to inf)
- [ ] reduction operations
- [ ] matrix multiplication
- [x] reshape
- [x] unary operation tests
- [ ] Display trait for tensors

July 19, 2026
- need to add operation overloading (+, -, *, /)
