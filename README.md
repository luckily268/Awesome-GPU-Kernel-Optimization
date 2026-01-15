# Awesome GPU Kernel Optimization
This is the repository of survey on Deep Learning Kernel Generation. For details, please refer to:
A Survey on Deep Learning Kernel Generation Using Large Language Models [paper]

## 🧩 Overview

This repository collects key research works, frameworks, and open-source projects related to **GPU kernel optimization**, **automatic tuning**, and **AI-based code generation**.
- **After LLMs (Post-LLM Era):** characterized by AI-assisted code generation, autonomous kernel synthesis, and data-driven performance tuning.

> Each entry is formatted as:
> **[Source] – [Paper] – [Repo]**
>
> - Source: arXiv / Conference / Blog / Tech Report  
> - Paper: paper title (linked)  
> - Code: GitHub / HF / Official repo if available, otherwise marked as 🔒

## 📚 Contents
📅 Last update on 2026/1/11
### 🔥🔥🔥Post-LLM Era: AI-based and LLM-driven Optimization
#### 🤖Single-Agent Systems
- arxiv'25-[KernelBench: Can LLMs Write Efficient GPU Kernels?](https://arxiv.org/abs/2502.10517)– 🔓 [![GitHub stars](https://img.shields.io/github/stars/ScalingIntelligence/KernelBench?style=social)](https://github.com/ScalingIntelligence/KernelBench)
- blog-[Automating GPU Kernel Generation with DeepSeek-R1 and Inference Time Scaling](https://developer.nvidia.com/blog/automating-gpu-kernel-generation-with-deepseek-r1-and-inference-time-scaling/)– 🔒
- Euro-Par'25 (LNCS)-[Tutoring LLM into a Better CUDA Optimizer](https://link.springer.com/chapter/10.1007/978-3-031-99857-7_18)– 🔓 [![GitHub stars](https://img.shields.io/github/stars/matyas-brabec/2025-europar-llm?style=social)](https://github.com/matyas-brabec/2025-europar-llm)
- arxiv'25-[GPU Performance Portability needs Autotuning](https://arxiv.org/abs/2505.03780)– 🔒
- arxiv'25-[CUDA-LLM:LLMs Can Write Efficient CUDA Kernels](https://arxiv.org/abs/2506.09092)– 🔒
- arxiv'25=[TritonForge: Profiling-Guided Framework for Automated Triton Kernel Optimization](https://arxiv.org/abs/2512.09196)– 🔓 [![GitHub stars](https://img.shields.io/github/stars/RLsys-Foundation/TritonForge?style=social)](https://github.com/RLsys-Foundation/TritonForge)
- arxiv'25-[EVOENGINEER: MASTERING AUTOMATED CUDA KERNEL CODE EVOLUTION WITH LARGE LANGUAGE MODELS](https://arxiv.org/pdf/2510.03760)– 🔒
- arvix'25-[From Large to Small: Transferring CUDA Optimization Expertise via Reasoning Graph](https://arxiv.org/abs/2510.19873)– 🔓 [![GitHub stars](https://img.shields.io/github/stars/blacknickwield/ReGraphT?style=social)](https://github.com/blacknickwield/ReGraphT)
- arxiv'25-[KernelBand: Boosting LLM-based Kernel Optimization with a Hierarchical and Hardware-aware Multi-armed Bandit](https://arxiv.org/abs/2511.18868)– 🔒


#### 🤖🤖Multi-Agent Systems
- arxiv'25-[GPU Kernel Scientist: An LLM-Driven Framework for Iterative Kernel Optimization](https://arxiv.org/abs/2506.20807)
- arxiv'25-[Geak: Introducing Triton Kernel AI Agent & Evaluation Benchmarks](https://arxiv.org/abs/2507.23194)– 🔓 (GEAK agent implementation)[![GitHub stars](https://img.shields.io/github/stars/AMD-AIG-AIMA/GEAK-agent?style=social)](https://github.com/AMD-AIG-AIMA/GEAK-agent)   (GEAK evaluation framework)[![GitHub stars](https://img.shields.io/github/stars/AMD-AIG-AIMA/GEAK-eval?style=social)](https://github.com/AMD-AIG-AIMA/GEAK-eval)
- blog-[How Many Agents Does it Take to Beat PyTorch?(surprisingly not that much)](https://letters.lossfunk.com/p/how-many-agents-does-it-take-to-beat)
- arxiv'25-[Astra: A multi-agent system for gpu kernel performance optimization](https://arxiv.org/abs/2509.07506)– 🔓 [![GitHub stars](https://img.shields.io/github/stars/Anjiang-Wei/Astra?style=social)](https://github.com/Anjiang-Wei/Astra)
- arxiv'25-[STARK:StrategicTeamofAgentsforRefining Kernels](https://arxiv.org/abs/2510.16996)– 🔒
- arxiv'25-[CudaForge: An Agent Framework with Hardware Feedback for CUDA Kernel Optimization](https://arxiv.org/abs/2511.01884)– 🔓 [![GitHub stars](https://img.shields.io/github/stars/OptimAI-Lab/CudaForge?style=social)](https://github.com/OptimAI-Lab/CudaForge)
- arxiv'25-[KForge: Program Synthesis for Diverse AI Hardware Accelerators](https://arxiv.org/abs/2511.13274)– 🔒
- arxiv'25-[AKG KERNEL AGENT: A MULTI-AGENT FRAMEWORK FOR CROSS-PLATFORM KERNEL SYNTHESIS](https://arxiv.org/pdf/2512.23424v1)– 🔓 (https://atomgit.com/mindspore/akg)
- Technical report, Sakana AI, 02 2025-[The AI CUDA engineer: Agentic CUDA kernel discovery, optimization and composition](https://pub.sakana.ai/static/paper.pdf)– 🔒
- KernelFalcon-[KernelFalcon: Autonomous GPU Kernel Generation via Deep Agents](https://pytorch.org/blog/kernelfalcon-autonomous-gpu-kernel-generation-via-deep-agents/)– 🔓 [![GitHub stars](https://img.shields.io/github/stars/meta-pytorch/KernelAgent?style=social)](https://github.com/meta-pytorch/KernelAgent)
- arxiv'25-[Optimizing PyTorch Inference with LLM-Based Multi-Agent Systems](https://arxiv.org/abs/2511.16964)– 🔒
- arxiv'25-[PRAGMA: A Profiling-Reasoned Multi-Agent Framework for Automatic Kernel Optimization](https://arxiv.org/abs/2511.06345)– 🔒

#### 🎯Training-based Methods
- arxiv'25(special for CUDA)-[Kevin: Multi-turn rl for generating cuda kernels](https://arxiv.org/abs/2507.11948)– 🔓 (https://huggingface.co/cognition-ai/Kevin-32B/tree/main)
- huggingface-[KernelLLM](https://huggingface.co/facebook/KernelLLM)
- arxiv'25-[Cuda-l1: Improving cuda optimization via contrastive reinforcement learning](https://arxiv.org/abs/2507.14111)– 🔓 [![GitHub stars](https://img.shields.io/github/stars/deepreinforce-ai/CUDA-L1?style=social)](https://github.com/deepreinforce-ai/CUDA-L1) [Project Page](https://deepreinforce-ai.github.io/cudal1_blog/)
- arxiv'25(special for Triton)-[AutoTriton: Automatic Triton Programming with Reinforcement Learning in LLMs](https://arxiv.org/abs/2507.05687)– 🔓 [![GitHub stars](https://img.shields.io/github/stars/AI9Stars/AutoTriton?style=social)](https://github.com/AI9Stars/AutoTriton)
- arxiv'25(special for Triton)-[TRITONRL: TRAINING LLMS TO THINK AND CODE TRITON WITHOUT CHEATING](https://arxiv.org/abs/2510.17891)– 🔓 (Codes and datasets will be available soon)
- arxiv'25-[QiMeng-Kernel: Macro-Thinking Micro-Coding Paradigm for LLM-Based High-Performance GPU Kernel Generation](https://arxiv.org/abs/2511.20100)– 🔓 [![GitHub stars](https://img.shields.io/github/stars/QiMeng-IPRC/QiMeng-Kernel?style=social)](https://github.com/QiMeng-IPRC/QiMeng-Kernel)
- CGO'25-[CuAsmRL: Optimizing GPU SASS Schedules via Deep Reinforcement Learning](https://dl.acm.org/doi/abs/10.1145/3696443.3708943)– 🔒
- underview-[Mastering Sparse {CUDA} Generation through Pretrained Models and Deep Reinforcement Learning](https://openreview.net/forum?id=VdLEaGPYWT)– 🔒
- arxiv'25-[SwizzlePerf: Hardware-Aware LLMs for GPU Kernel Performance Optimization](https://arxiv.org/abs/2508.20258)– 🔒
- arxiv'25-[Integrating Performance Tools in Model Reasoning for GPU Kernel Optimization](https://arxiv.org/abs/2510.17158)– 🔒

#### 📱Mobile
- arxiv'25-[Scaling LLM Test-Time Compute with Mobile NPU on Smartphones](https://arxiv.org/abs/2509.23324)– 🔓 (main
repo)[![GitHub stars](https://img.shields.io/github/stars/haozixu/llama.cpp-npu?style=social)](https://github.com/haozixu/llama.cpp-npu) and (op library)[![GitHub stars](https://img.shields.io/github/stars/haozixu/htp-ops-lib?style=social)](https://github.com/haozixu/htp-ops-lib) 


#### 💹Benchmark Suites
- arxiv'25-[KernelBench: Can LLMs Write Efficient GPU Kernels?](https://arxiv.org/abs/2502.10517)– 🔓 [![GitHub stars](https://img.shields.io/github/stars/ScalingIntelligence/KernelBench?style=social)](https://github.com/ScalingIntelligence/KernelBench)
- arxiv'25-[MultiKernelBench: A Multi-Platform Benchmark for Kernel Generation](https://arxiv.org/abs/2507.17773)– 🔓 [![GitHub stars](https://img.shields.io/github/stars/wzzll123/MultiKernelBench?style=social)](https://github.com/wzzll123/MultiKernelBench)
- HPDX'25-[Can Large Language Models Predict Parallel Code Performance](https://dl.acm.org/doi/abs/10.1145/3731545.3743645)– 🔓 [![GitHub stars](https://img.shields.io/github/stars/Scientific-Computing-Lab/ParallelCodeEstimation?style=social)](https://github.com/Scientific-Computing-Lab/ParallelCodeEstimation)
- arxiv'25-[NPUEval: Optimizing NPU Kernels with LLMs and Open Source Compilers](https://arxiv.org/abs/2507.14403)– 🔓 [![GitHub stars](https://img.shields.io/github/stars/Scientific-Computing-Lab/ParallelCodeEstimation?style=social)](https://github.com/Scientific-Computing-Lab/ParallelCodeEstimation)– 🔓 [![GitHub stars](https://img.shields.io/github/stars/AMDResearch/NPUEval?style=social)](https://github.com/AMDResearch/NPUEval)
- arxiv'25-[Towards robust agentic cuda kernel benchmarking, verification, and optimization](https://arxiv.org/abs/2509.14279)– 🔓 [![GitHub stars](https://img.shields.io/github/stars/SakanaAI/robust-kbench?style=social)](https://github.com/SakanaAI/robust-kbench)
- github-[BackendBanch](https://github.com/meta-pytorch/BackendBench)
- arxiv.25'-[ConCuR: Conciseness Makes State-of-the-Art Kernel Generation](https://arxiv.org/abs/2510.07356)– 🔓 (https://huggingface.co/lkongam/KernelCoder)
- arxiv'26 - [FLASHINFER-BENCH: BUILDING THE VIRTUOUS CYCLE FOR AI-DRIVEN LLM SYSTEMS](https://arxiv.org/abs/2601.00227)– 🔓 [![GitHub stars](https://img.shields.io/github/stars/flashinfer-ai/flashinfer-bench?style=social)](https://github.com/flashinfer-ai/flashinfer-bench)

---

## 🔍 Related Surveys

| Title | Year |
|-------|------|
| [Auto-tuning of GPU Kernels: Techniques and Trends](https://dl.acm.org/doi/10.1145/3650200.3656626)– 🔓 [![GitHub stars](https://img.shields.io/github/stars/HPCRL/Ansor-AF-DS?style=social)](https://github.com/HPCRL/Ansor-AF-DS) | 2020 |
| [A Review of CUDA Optimization Techniques and Tools for Structured Grid Computing](https://link.springer.com/article/10.1007/s00607-019-00744-1)| 2019 |
| [A Survey on Large Language Models for Code Generation](https://arxiv.org/abs/2406.00515)– 🔓 [![GitHub stars](https://img.shields.io/github/stars/juyongjiang/CodeLLMSurvey?style=social)](https://github.com/juyongjiang/CodeLLMSurvey) | 2024 | 



---

## 📖 Common Acronyms (Awesome GPU Kernel Optimization)

| Acronym | Full Form | Scope |
|--------|----------|------|
| AKG | Automatic Kernel Generator | MindSpore / TVM-style kernel generators |
| AST | Abstract Syntax Tree | Code representation |
| BK | Benchmark / Benchmarking | Dataset & eval suites |
| BT | Block Tiling | Kernel tiling strategy |
| CTA | Cooperative Thread Array | CUDA execution unit |
| CUDA | Compute Unified Device Architecture | NVIDIA GPU programming |
| DCE | Dead Code Elimination | Compiler optimization |
| DSL | Domain-Specific Language | Triton / TVM IR |
| FLOPs | Floating-Point Operations | Performance metric |
| GPU | Graphics Processing Unit | Target hardware |
| HF | HuggingFace | Model & dataset hosting |
| ILP | Instruction-Level Parallelism | Micro-architecture metric |
| ISA | Instruction Set Architecture | PTX / SASS level |
| LLM | Large Language Model | Kernel generation agents |
| MA | Multi-Agent System | Agent frameworks |
| MMA | Matrix Multiply Accumulate | Tensor Core primitive |
| NCU | NVIDIA Compute Utility | GPU profiler |
| NPU | Neural Processing Unit | Mobile / edge accelerators |
| OI | Operational Intensity | Roofline metric |
| OSS | Open-Source Software | Reproducibility |
| PTX | Parallel Thread Execution | CUDA IR |
| RL | Reinforcement Learning | Training-based methods |
| SA | Single-Agent System | Kernel synthesis agent |
| SASS | Streaming Assembler | NVIDIA machine code |
| SM | Streaming Multiprocessor | GPU core unit |
| TB | Training-Based Method | RL / fine-tuned LLMs |
| TMA | Tensor Memory Accelerator | Hopper architecture |
| Triton | Tensor IR Toolchain for Operators | Kernel DSL |

---

### 🤖 Agent & Optimization Frameworks

| Acronym | Full Form |
|--------|----------|
| FSR | Failure-Sensitive Revision |
| SDS | Search-Driven Synthesis |
| ISO | Iterative Search Optimization |
| PRM | Process Reward Model |
| ORM | Outcome Reward Model |

---

### 💹 Benchmarks & Evaluation

| Acronym | Full Form |
|--------|----------|
| KBench | KernelBench |
| MKB | MultiKernelBench |
| NPU-Eval | Neural Processing Unit Evaluation |
| Roofline | Roofline Performance Model |


## 👋Contributing
Contributions are welcome! If you have new resources, tools, or insights to add, feel free to submit a pull request.
For detailed guidelines, please refer to [CONTRIBUTING.md](./CONTRIBUTING.md).

---

## 📄 License

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

© 2025 Awesome GPU Kernel Optimization. Licensed under the MIT License.  
See [LICENSE](./LICENSE).

## Citation

```bib
@article{DLKernel,
  title={Recent Advances in Deep Learning Kernel Optimization Using Large Language Models},
  author={Tianlin Li, Chenxi Yang, Qitong Sun, Xiaoyu Zhang, Qiang Hu, Zhe Tang, Sheng Chen, Fei Yang, Aishan Liu, Xianglong Liu, Chao Shen, Yves Le Traon, Yang Liu},
  year={2025}
  url={https://github.com/luckily268/Awesome-GPU-Kernel-Optimization}
}
```
