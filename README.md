# Awesome-GPU-Kernel-Optimization
A curated list of resources on GPU kernel optimization — from manual tuning to AI-driven code generation with LLMs.

---

## 🧩 Overview

This repository collects key research works, frameworks, and open-source projects related to **GPU kernel optimization**, **automatic tuning**, and **AI-based code generation**.

It aims to provide a clear picture of how GPU kernel optimization evolved:
- **Before LLMs (Pre-LLM Era):** dominated by manual tuning, rule-based optimization, and compiler-guided heuristics.
- **After LLMs (Post-LLM Era):** characterized by AI-assisted code generation, autonomous kernel synthesis, and data-driven performance tuning.

---

## 📚 Contents

### 1. Pre-LLM Era: Manual and Compiler-based Optimization
- Kernel auto-tuning frameworks (e.g., AutoTVM, Halide, Ansor)
- Analytical performance modeling and cost estimation
- Static optimization and hardware-aware scheduling
- Operator-specific optimization (e.g., GEMM, convolution, reduction)

### 2. Post-LLM Era: AI-based and LLM-driven Optimization

#### SFT+RL
- arxiv'25-[Cuda-l1: Improving cuda optimization via contrastive reinforcement learning](https://arxiv.org/abs/2507.14111)
- arxiv'25-[CUDA-LLM:LLMs Can Write Efficient CUDA Kernels](https://arxiv.org/abs/2506.09092)
- arxiv'25(special for Triton)-[AutoTriton: Automatic Triton Programming with Reinforcement Learning in LLMs](https://arxiv.org/abs/2507.05687)
- arxiv'25(special for CUDA)-[Kevin: Multi-turn rl for generating cuda kernels](https://arxiv.org/abs/2507.11948)
- arxiv'25(special for Triton)-[TRITONRL: TRAINING LLMS TO THINK AND CODE TRITON WITHOUT CHEATING](https://arxiv.org/abs/2510.17891)


#### Hareware and Compiler
- IEEE'24-[STuning-DL: Model-Driven Autotuning of Sparse GPU Kernels for Deep Learning](https://ieeexplore.ieee.org/abstract/document/10534045/)
- arxiv'24-[Thunderkittens: Simple, fast, and adorable ai kernels](https://arxiv.org/abs/2410.20399)
- arxiv'25-[SwizzlePerf: Hardware-Aware LLMs for GPU Kernel Performance Optimization](https://arxiv.org/abs/2508.20258)
- arxiv'25-[Geak: Introducing Triton Kernel AI Agent & Evaluation Benchmarks](https://arxiv.org/abs/2507.23194)
- arxiv'25-[GPU Performance Portability needs Autotuning](https://arxiv.org/abs/2505.03780)
- arxiv'25-[Scaling LLM Test-Time Compute with Mobile NPU on Smartphones](https://arxiv.org/abs/2509.23324)
- arxiv'25-[A Few Fit Most: Improving Performance Portability of SGEMM on GPUs using Multi-Versioning](https://arxiv.org/abs/2507.15277)


#### 🧠 Memory
- JMLR'21-[Kernel operations on the gpu, with autodiff, without memory overflows](https://www.jmlr.org/papers/v22/20-275.html)
- IEEE'21-[Bayesian Optimization for auto-tuning GPU kernels](https://ieeexplore.ieee.org/abstract/document/9652797/)


#### ⚙️ Algorithm
- IEEE'22-[Benchmarking optimization algorithms for auto-tuning GPU kernels](https://ieeexplore.ieee.org/abstract/document/9905732/)
- arxiv'25-[EVOENGINEER: MASTERING AUTOMATED CUDA KERNEL CODE EVOLUTION WITH LARGE LANGUAGE MODELS](https://arxiv.org/pdf/2510.03760)


#### 🧩 Intermediate Representation (IR)
- IEEE'21-[Reverse-Mode Automatic Differentiation and Optimization of GPU Kernels Via Enzyme](https://ieeexplore.ieee.org/document/9652797)
- CGO'25-[Proteus: Portable Runtime Optimization of GPU Kernel Execution with Just-in-Time Compilation](https://dl.acm.org/doi/abs/10.1145/3696443.3708939)
- arxiv'25-[KPerfIR: Towards an Open and Compiler-centric Ecosystem for GPU Kernel Performance Tooling on Modern AI Workloads](https://arxiv.org/abs/2505.21661)
- CGO'25-[CuAsmRL: Optimizing GPU SASS Schedules via Deep Reinforcement Learning](https://dl.acm.org/doi/abs/10.1145/3696443.3708943)
- arxiv'25-[Integrating Performance Tools in Model Reasoning for GPU Kernel Optimization](https://arxiv.org/abs/2510.17158)


#### 🤖 Agent / AI-based Optimization
- arxiv'25-[GPU Kernel Scientist: An LLM-Driven Framework for Iterative Kernel Optimization](https://arxiv.org/abs/2506.20807)
- PLDI'21-[DeepCuts: a Deep Learning Optimization Framework for Versatile GPU Workloads](https://dl.acm.org/doi/abs/10.1145/3453483.3454038)
- Euro-Par'25 (LNCS)-[Tutoring LLM into a Better CUDA Optimizer](https://link.springer.com/chapter/10.1007/978-3-031-99857-7_18)
- arxiv'25-[STARK:StrategicTeamofAgentsforRefining Kernels](https://arxiv.org/abs/2510.16996)
- arxiv'25-[CudaForge: An Agent Framework with Hardware Feedback for CUDA Kernel Optimization](https://arxiv.org/abs/2511.01884)

#### cache
- IEEE'24-[Pushing the Performance Envelope of DNN-based Recommendation Systems Inference on GPUs](https://ieeexplore.ieee.org/abstract/document/10764622/)



#### Blog
- Multi-agent-[How Many Agents Does it Take to Beat PyTorch?(surprisingly not that much)](https://letters.lossfunk.com/p/how-many-agents-does-it-take-to-beat)
- KernelFalcon-[KernelFalcon: Autonomous GPU Kernel Generation via Deep Agents](https://pytorch.org/blog/kernelfalcon-autonomous-gpu-kernel-generation-via-deep-agents/)
- Deepseek-R1(NVDIA DEVELOPER)-[Automating GPU Kernel Generation with DeepSeek-R1 and Inference Time Scaling](https://developer.nvidia.com/blog/automating-gpu-kernel-generation-with-deepseek-r1-and-inference-time-scaling/)



#### Benchmark Suites & Datasets
- arxiv'25-[KernelBench: Can LLMs Write Efficient GPU Kernels?](https://arxiv.org/abs/2502.10517)
- arxiv'25-[MultiKernelBench: A Multi-Platform Benchmark for Kernel Generation](https://arxiv.org/abs/2507.17773)
- HPDX'25-[Can Large Language Models Predict Parallel Code Performance](https://dl.acm.org/doi/abs/10.1145/3731545.3743645)
- arxiv'25-[The ai cuda engineer: Agentic cuda kernel discovery, optimization and composition（Towards Robust Agentic CUDA Kernel Benchmarking, Verification, and Optimization）](https://pub.sakana.ai/static/paper.pdf)
- arxiv.25'-[ConCuR: Conciseness Makes State-of-the-Art Kernel Generation](https://arxiv.org/abs/2510.07356)


---

## 🔍 Related Surveys

| Title | Year | Focus |
|-------|------|--------|
| “Auto-tuning of GPU Kernels: Techniques and Trends” | 2020 | [Pre-LLM tuning methodologies](https://dl.acm.org/doi/10.1145/3650200.3656626) |
