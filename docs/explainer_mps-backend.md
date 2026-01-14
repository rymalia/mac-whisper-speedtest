# EXPLAINER: APPLE MPS BACKEND

The **Apple MPS backend** (Metal Performance Shaders) is **a framework that allows machine learning (ML) libraries, such as PyTorch and TensorFlow, to use the integrated GPU in Apple Silicon (M1, M2, M3, M4 chips, etc.) and some Intel-based Macs for accelerated computation**. 

## Key Concepts

* **Purpose:** The primary goal of the MPS backend is to enable high-performance training and inference of ML models on macOS, leveraging Apple's Metal graphics API for GPU access.  
* **Alternative to CUDA:** It functions as Apple's specific alternative to NVIDIA's widely used CUDA platform. Developers use `mymodel.to("mps")` in PyTorch, similar to how they would use `mymodel.to("cuda")` on NVIDIA systems, to move computations from the CPU to the GPU.  
* **Optimized Kernels:** The framework contains a collection of highly optimized compute shaders (small programs that run on the GPU) that are fine-tuned for the unique hardware characteristics of Apple's GPUs. This ensures optimal performance without developers needing to write custom code for each GPU generation.  
* **Integration with ML Frameworks:** MPS provides a device and a graph framework (MPS Graph) that maps ML computational graphs and primitives to these efficient kernels. This allows higher-level frameworks like PyTorch and TensorFlow to utilize Apple's hardware efficiently.
* **Foundation for Higher-Level Frameworks:** MPS serves as the underlying GPU acceleration layer that Apple's MLX framework builds upon. While developers can use MPS directly via PyTorch (`device="mps"`), MLX abstracts this away entirely, providing automatic unified memory management. See the [MLX Framework Explainer](./explainer_mlx-framework.md) for details.

## How It Works (in PyTorch)

**When a developer specifies the "mps" device in their PyTorch code, the framework performs the following actions:**

1. **Maps Operations:** PyTorch operations (like matrix multiplication or convolutions) are mapped to the high-performance kernels within the MPS Graph framework.  
2. **Reduces Overhead:** The Metal API is designed to minimize the overhead of transferring data between the CPU and GPU, which further enhances efficiency.  
3. **Parallel Computation:** Deep learning tasks involve millions of parallel calculations across large datasets (tensors). The MPS backend leverages the GPU's numerous cores to handle these tasks in parallel, offering significant speedups compared to running on the CPU. 


## Advantages and Limitations

* **Performance:** Offers a significant speed boost (10x-20x) over CPU-only training on Apple Silicon.  
* **Hardware Efficiency:** Maximizes the utilization of the unified memory architecture in Apple Silicon chips, reducing data transfer bottlenecks.  
* **Limitations:**  
  * **Memory Management:** The MPS backend can have stricter memory buffer size limits than CUDA, which may lead to out-of-memory errors when processing extremely large sequences (e.g., in some Large Language Model tasks), requiring specific optimization techniques like chunking.  
  * **Operator Coverage:** While support is continually improving (refer to the PyTorch documentation on MPS), some less common operations may not yet be fully implemented, which can cause fallback to the CPU or errors if not handled correctly. 

## References

* [Accelerated PyTorch training on Mac with MPS (2023)](https://developer.apple.com/metal/pytorch/)
* [Getting started with Apple MLX (2025)](https://wandb.ai/byyoung3/ML_NEWS3/reports/Getting-started-with-Apple-MLX--Vmlldzo5Njk5MTk1)
* [Apple Developer: Metal Performance Shaders](https://developer.apple.com/documentation/metalperformanceshaders) — Foundational API documentation; MPS has evolved significantly since its introduction, with ongoing improvements to operator coverage and performance.
