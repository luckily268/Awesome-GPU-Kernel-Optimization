# Awesome GPU Kernel Optimization
This is the repository of survey on Deep Learning Kernel Generation. For details, please refer to:
A Survey on Deep Learning Kernel Generation Using Large Language Models [paper]

## 🧩 Overview

This repository collects key research works, frameworks, and open-source projects related to **GPU kernel optimization**, **automatic tuning**, and **AI-based code generation**.

It aims to provide a clear picture of how GPU kernel optimization evolved:
- **Before LLMs (Pre-LLM Era):** dominated by manual tuning, rule-based optimization, and compiler-guided heuristics.
- **After LLMs (Post-LLM Era):** characterized by AI-assisted code generation, autonomous kernel synthesis, and data-driven performance tuning.


## 📚 Contents

### 1. 🚂Pre-LLM Era: Manual and Compiler-based Optimization

#### 📚Journal Article
- ACM Comput. Surv., Vol. 55, No. 11-[Optimization Techniques for GPU Programming](https://dl.acm.org/doi/full/10.1145/3570638)
- Mach. Vis. Appl.'13-[An optimized approach to histogram computation on GPU](https://link.springer.com/article/10.1007/s00138-012-0443-3)
- Concurr. Comput.-P. E.'09-[Exploiting graphical processing units for data-parallel scientific applications](https://onlinelibrary.wiley.com/doi/abs/10.1002/cpe.1462)
- Comput. Electr. Eng.'20-[Accelerating sparse matrix–matrix multiplication with GPU Tensor Cores](https://www.sciencedirect.com/science/article/pii/S0045790620307011)
- Int. J. High Perform. C.'10-[An improved magma gemm for fermi graphics processing units](https://journals.sagepub.com/doi/abs/10.1177/1094342010385729)
- IEEE-TPDS'13-[Medusa: Simplified graph processing on GPUs](https://ieeexplore.ieee.org/abstract/document/6497047/)
- Procedia Comput. Sci.'16-[Performance tuning and optimization techniques of fixed and variable size batched Cholesky factorization on GPUs](https://www.sciencedirect.com/science/article/pii/S1877050916306548)
- Parallel Comput.'18-[Benchmarking the GPU memory at the warp level](https://www.sciencedirect.com/science/article/abs/pii/S0167819117301825)
- Future Gener. Comp. Sy.'20-[A benchmark set of highly-efficient CUDA and OpenCL kernels and its dynamic autotuning with Kernel Tuning Toolkit](https://www.sciencedirect.com/science/article/abs/pii/S0167739X19327360)
- ACM-TOMS'20-[Strassen’s algorithm reloaded on GPUs](https://dl.acm.org/doi/abs/10.1145/3372419)
- Concurr. Comput.-P. E.'18-[Accelerating explicit ODE methods on GPUs by kernel fusion](https://onlinelibrary.wiley.com/doi/abs/10.1002/cpe.4470)
- GeoInformatica'18-[A compiler approach to map algebra: automatic parallelization, locality optimization, and GPU acceleration of raster spatial analysis](https://link.springer.com/article/10.1007/s10707-017-0312-3)
- J. Inf. Sci. Eng.'16-[GPU-Based High Performance Password Recovery Technique for Hash Functions](https://www.academia.edu/115050262/GPU_Based_High_Performance_Password_Recovery_Technique_for_Hash_Functions)
- Sci. Program.'18-[A strategy for automatic performance tuning of stencil computations on GPUs](https://onlinelibrary.wiley.com/doi/full/10.1155/2018/6093054)
- arXiv'25-[Accelerating Bangla NLP Tasks with Automatic Mixed Precision: Resource-Efficient Training Preserving Model Efficacy](https://arxiv.org/abs/2512.00829)
- arXiv'24-[Evaluating Quantized Large Language Models for Code Generation on Low-Resource Language Benchmarks](https://www.sciencedirect.com/science/article/abs/pii/S2590118425000371)
- arXiv'25-[Compiler-supported reduced precision and AoS-SoA transformations for heterogeneous hardware](https://arxiv.org/abs/2512.05516)
- arXiv'23-[Execution-based Code Generation using Deep Reinforcement Learning](https://arxiv.org/abs/2301.13816)
- arXiv'22-[Compilable Neural Code Generation with Compiler Feedback](https://arxiv.org/abs/2203.05132)
- arXiv'23-[RLTF: Reinforcement Learning from Unit Test Feedback](https://arxiv.org/abs/2307.04349)
- IEEE/ACM'29-[Style-Analyzer: Fixing Code Style Inconsistencies with Interpretable Unsupervised Algorithms](https://ieeexplore.ieee.org/abstract/document/8816753/)
- IEEE'09-[Learning a metric for code readability](https://ieeexplore.ieee.org/abstract/document/5332232)



#### 🎤Conference Paper
- SC'18-[Tricore: Parallel triangle counting on gpus](https://ieeexplore.ieee.org/abstract/document/8665796)
- SC'11-[Optimizing symmetric dense matrix-vector multiplication on GPUs](https://dl.acm.org/doi/abs/10.1145/2063384.2063392)
- ICS'12-[High-performance code generation for stencil computations on GPU architectures](https://dl.acm.org/doi/abs/10.1145/2304576.2304619)
- IISWC'19-[SNU-NPB 2019: parallelizing and optimizing NPB in OpenCL and CUDA for modern GPUs](https://ieeexplore.ieee.org/abstract/document/9041954)
- HiPC'15-[Memory-efficient parallelization of 3D lattice Boltzmann flow solver on a GPU](https://ieeexplore.ieee.org/abstract/document/7397646/)
- ICS'16-[Balanced hashing and efficient gpu sparse general matrix-matrix multiplication](https://dl.acm.org/doi/abs/10.1145/2925426.2926273)
- HiPC'12-[Sparse matrix-matrix multiplication on modern architectures](https://ieeexplore.ieee.org/abstract/document/6507483)
- Computing Conf.'17-[High performance CUDA AES implementation: A quantitative performance analysis approach](https://ieeexplore.ieee.org/abstract/document/8252225)
- SAMOS'10-[Compile-time GPU memory access optimizations](https://ieeexplore.ieee.org/abstract/document/5642066)
- SC'11-[CudaDMA: optimizing GPU memory bandwidth via warp specialization](https://dl.acm.org/doi/abs/10.1145/2063384.2063400)
- CANDAR'15-[A warp-synchronous implementation for multiple-length multiplication on the GPU](https://ieeexplore.ieee.org/abstract/document/7424695)
- IPDPS'20-[Demystifying tensor cores to optimize half-precision matrix multiply](https://ieeexplore.ieee.org/abstract/document/9139835)
- NeurIPS'22-[CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/8636419dea1aa9fbd25fc4248e702da4-Abstract-Conference.html)

### 2. 🔥🔥🔥Post-LLM Era: AI-based and LLM-driven Optimization
#### 🤖Single-Agent Systems
- arxiv'25-[KernelBench: Can LLMs Write Efficient GPU Kernels?](https://arxiv.org/abs/2502.10517)
- blog-[Automating GPU Kernel Generation with DeepSeek-R1 and Inference Time Scaling](https://developer.nvidia.com/blog/automating-gpu-kernel-generation-with-deepseek-r1-and-inference-time-scaling/)
- Euro-Par'25 (LNCS)-[Tutoring LLM into a Better CUDA Optimizer](https://link.springer.com/chapter/10.1007/978-3-031-99857-7_18)
- arxiv'25-[GPU Performance Portability needs Autotuning](https://arxiv.org/abs/2505.03780)
- arxiv'25-[CUDA-LLM:LLMs Can Write Efficient CUDA Kernels](https://arxiv.org/abs/2506.09092)
- arxiv'25=[TritonForge: Profiling-Guided Framework for Automated Triton Kernel Optimization](https://arxiv.org/abs/2512.09196)
- arxiv'25-[EVOENGINEER: MASTERING AUTOMATED CUDA KERNEL CODE EVOLUTION WITH LARGE LANGUAGE MODELS](https://arxiv.org/pdf/2510.03760)
- arvix'25-[From Large to Small: Transferring CUDA Optimization Expertise via Reasoning Graph](https://arxiv.org/abs/2510.19873)
- arxiv'25-[KernelBand: Boosting LLM-based Kernel Optimization with a Hierarchical and Hardware-aware Multi-armed Bandit](https://arxiv.org/abs/2511.18868)


#### 🤖🤖Multi-Agent Systems
- arxiv'25-[GPU Kernel Scientist: An LLM-Driven Framework for Iterative Kernel Optimization](https://arxiv.org/abs/2506.20807)
- arxiv'25-[Geak: Introducing Triton Kernel AI Agent & Evaluation Benchmarks](https://arxiv.org/abs/2507.23194)
- blog-[How Many Agents Does it Take to Beat PyTorch?(surprisingly not that much)](https://letters.lossfunk.com/p/how-many-agents-does-it-take-to-beat)
- arxiv'25-[Astra: A multi-agent system for gpu kernel performance optimization](https://arxiv.org/abs/2509.07506)
- arxiv'25-[STARK:StrategicTeamofAgentsforRefining Kernels](https://arxiv.org/abs/2510.16996)
- arxiv'25-[CudaForge: An Agent Framework with Hardware Feedback for CUDA Kernel Optimization](https://arxiv.org/abs/2511.01884)
- arxiv'25-[KForge: Program Synthesis for Diverse AI Hardware Accelerators](https://arxiv.org/abs/2511.13274)
- github-[AKG](https://atomgit.com/mindspore/akg)
- Technical report, Sakana AI, 02 2025-[The AI CUDA engineer: Agentic CUDA kernel discovery, optimization and composition](https://pub.sakana.ai/static/paper.pdf)
- KernelFalcon-[KernelFalcon: Autonomous GPU Kernel Generation via Deep Agents](https://pytorch.org/blog/kernelfalcon-autonomous-gpu-kernel-generation-via-deep-agents/)
- arxiv'25-[Optimizing PyTorch Inference with LLM-Based Multi-Agent Systems](https://arxiv.org/abs/2511.16964)
- arxiv'25-[PRAGMA: A Profiling-Reasoned Multi-Agent Framework for Automatic Kernel Optimization](https://arxiv.org/abs/2511.06345)

#### 🎯Training-based Methods
- arxiv'25(special for CUDA)-[Kevin: Multi-turn rl for generating cuda kernels](https://arxiv.org/abs/2507.11948)
- huggingface-[KernelLLM](https://huggingface.co/facebook/KernelLLM)
- arxiv'25-[Cuda-l1: Improving cuda optimization via contrastive reinforcement learning](https://arxiv.org/abs/2507.14111)
- arxiv'25(special for Triton)-[AutoTriton: Automatic Triton Programming with Reinforcement Learning in LLMs](https://arxiv.org/abs/2507.05687)
- arxiv'25(special for Triton)-[TRITONRL: TRAINING LLMS TO THINK AND CODE TRITON WITHOUT CHEATING](https://arxiv.org/abs/2510.17891)
- arxiv'25-[QiMeng-Kernel: Macro-Thinking Micro-Coding Paradigm for LLM-Based High-Performance GPU Kernel Generation](https://arxiv.org/abs/2511.20100)
- CGO'25-[CuAsmRL: Optimizing GPU SASS Schedules via Deep Reinforcement Learning](https://dl.acm.org/doi/abs/10.1145/3696443.3708943)
- underview-[Mastering Sparse {CUDA} Generation through Pretrained Models and Deep Reinforcement Learning](https://openreview.net/forum?id=VdLEaGPYWT)
- arxiv'25-[SwizzlePerf: Hardware-Aware LLMs for GPU Kernel Performance Optimization](https://arxiv.org/abs/2508.20258)
- arxiv'25-[Integrating Performance Tools in Model Reasoning for GPU Kernel Optimization](https://arxiv.org/abs/2510.17158)

#### 📱Mobile
- arxiv'25-[Scaling LLM Test-Time Compute with Mobile NPU on Smartphones](https://arxiv.org/abs/2509.23324)



#### 💹Benchmark Suites & Datasets
- arxiv'25-[KernelBench: Can LLMs Write Efficient GPU Kernels?](https://arxiv.org/abs/2502.10517)
- arxiv'25-[MultiKernelBench: A Multi-Platform Benchmark for Kernel Generation](https://arxiv.org/abs/2507.17773)
- HPDX'25-[Can Large Language Models Predict Parallel Code Performance](https://dl.acm.org/doi/abs/10.1145/3731545.3743645)
- arxiv'25-[NPUEval: Optimizing NPU Kernels with LLMs and Open Source Compilers](https://arxiv.org/abs/2507.14403)
- arxiv'25-[Towards robust agentic cuda kernel benchmarking, verification, and optimization](https://arxiv.org/abs/2509.14279)
- github-[BackendBanch](https://github.com/meta-pytorch/BackendBench)
- arxiv.25'-[ConCuR: Conciseness Makes State-of-the-Art Kernel Generation](https://arxiv.org/abs/2510.07356)


---

## 🔍 Related Surveys

| Title | Year |
|-------|------|
| [Auto-tuning of GPU Kernels: Techniques and Trends](https://dl.acm.org/doi/10.1145/3650200.3656626) | 2020 |
| [A Review of CUDA Optimization Techniques and Tools for Structured Grid Computing](https://link.springer.com/article/10.1007/s00607-019-00744-1)| 2019 |
| [A Survey on Large Language Models for Code Generation](https://arxiv.org/abs/2406.00515) | 2024 |
