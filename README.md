# Awesome GPU Kernel Optimization

A curated companion to **Recent Advances in Deep Learning Kernel Optimization Using Large Language Models**.

This repository catalogs LLM-based deep learning kernel generation and optimization: evaluation benchmarks, single-agent and multi-agent systems, training-based methods, and research directions across CUDA, high-level kernel DSLs and alternative accelerators.

**[Survey PDF](./Recent%20Advances%20in%20Deep%20Learning%20Kernel%20OptimizationUsing%20Large%20Language%20Models.pdf)** · **[Kernel Atlas — Interactive Literature Map](https://luckily268.github.io/kernel-atlas/)** · **[Contributing](./CONTRIBUTING.md)**

**README updated:** October 19, 2026  
**Survey search cutoff:** October 10, 2026  
**Study window:** February 1, 2025 – October 10, 2026, based on first public availability.

## Overview

The current survey includes **76 independent primary studies**: **70 scholarly papers/preprints and 6 standalone first-party technical studies**. Its catalog contains **16 benchmark studies** and **62 method studies**; AI CUDA Engineer / robust-kbench and AscendKernelGen / NPUKernelBench appear in both roles and are counted once in the 76-study corpus.

The survey addresses two research questions:

- **RQ1 — Evaluation:** What benchmarks have been proposed for LLM-based deep learning kernel optimization evaluation?
- **RQ2 — Methods:** What LLM-based approaches have been proposed for deep learning kernel optimization?

The roadmap combines their findings with established GPU optimization expertise, compiler techniques and advances in code language models.

### Reading the catalog

- Groupings follow the current survey: **six benchmark categories** and **four method categories**.
- For CUDA and high-level DSL methods, subcategories identify the primary methodological focus. Hardware and backend scope define the subcategories for alternative-accelerator and cross-platform methods. A training-based approach can also use multiple agents.
- Years follow the survey timeline, rather than the year of a later conference publication. Task counts describe the surveyed benchmark releases and may differ from newer upstream versions.
- Work names link to papers, technical posts or official project pages. **Resources** provides identified first-party code, model, dataset or project links. A dash means no additional link is listed here; it does not establish that a project is closed source. A repository link alone does not imply that the full method is released.
- Versioned links preserve surveyed names when later paper revisions change them, including KernelBand, ARGUS and KLineage. Distinct projects named **Kernel Foundry** and **KernelFoundry** are identified by author.

## Contents

- [Evaluation Benchmarks](#evaluation-benchmarks)
- [Kernel Generation and Optimization Techniques](#kernel-generation-and-optimization-techniques)
- [Evaluation Dimensions](#evaluation-dimensions)
- [Research Roadmap](#research-roadmap)
- [Related Surveys and Background](#related-surveys-and-background)
- [Common Terms](#common-terms)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## Evaluation Benchmarks

The six categories below follow Section 4.1 of the survey. Their boundaries reflect evaluation focus as well as programming abstraction: for example, D2K-Bench is discussed in the CUDA-centric group while evaluating Triton implementations on NVIDIA hardware.

### CUDA-Centric Benchmarks

| Year | Work | Focus | Resources |
|---|---|---|---|
| 2025 | [KernelBench](https://arxiv.org/abs/2502.10517) | PyTorch → CUDA; 270 tasks across L1–L4 in the surveyed release; correctness and speedup | [Code](https://github.com/ScalingIntelligence/KernelBench) |
| 2025 | [ComputeEval](https://github.com/NVIDIA/compute-eval) | 128 CUDA programming problems; functional testing and selected performance tasks | [Code](https://github.com/NVIDIA/compute-eval) |
| 2025 | [robust-kbench](https://arxiv.org/abs/2509.14279) | CUDA robustness across input shapes, initializations and forward/backward settings | [Benchmark](https://github.com/SakanaAI/robust-kbench) |
| 2026 | [CUDABench](https://arxiv.org/abs/2603.02236) | Text → CUDA; 500 tasks and 1,500 natural-language prompts | [Code](https://github.com/CUDA-Bench/CUDABench) |
| 2026 | [MSKernelBench](https://arxiv.org/abs/2603.07169) | 50 operators × FP32/BF16; multi-scenario, multi-scale CUDA evaluation | [Project](https://hanyx2021.github.io/MSKernelBenchDemo/) |
| 2026 | [PTXBench](https://arxiv.org/abs/2608.17379) | Architecture-specific inline PTX; GEMM/attention on H100 and B200; dynamic instruction counts | — |
| 2026 | [D2K-Bench](https://arxiv.org/abs/2610.03226) | 26 tasks / 85 workloads; expert-guided Triton generation and design/code assessment | — |

### High-Level Kernel DSL Benchmarks

| Year | Work | Focus | Resources |
|---|---|---|---|
| 2025 | [TritonBench](https://aclanthology.org/2025.findings-acl.1183/) | Triton operator generation; 350 tasks across the two channels summarized in the survey | — |
| 2025 | [BackendBench](https://github.com/meta-pytorch/BackendBench) | PyTorch backend correctness and performance under practical operator workloads | [Code](https://github.com/meta-pytorch/BackendBench) |
| 2026 | [KernelBenchX](https://arxiv.org/abs/2605.04956) | 176 tasks / 15 structural categories; correctness, robustness and hardware efficiency | [Code](https://github.com/BonnieW05/KernelBenchX) |

### Alternative Accelerator Benchmarks

| Year | Work | Focus | Resources |
|---|---|---|---|
| 2025 | [NPUEval](https://arxiv.org/abs/2507.14403) | Vectorized kernel generation for AMD NPUs; correctness and cycle-level performance | [Code](https://github.com/AMDResearch/NPUEval) |
| 2026 | [NPUKernelBench](https://arxiv.org/abs/2601.07160) | 158 AscendC tasks; device-only and host–device generation on Ascend NPUs | [Benchmark](https://github.com/weich97/NPUKernelBench) · [Models](https://huggingface.co/AscendKernelGen) |
| 2026 | [AgentKernelArena](https://arxiv.org/abs/2605.16819) | 196 HIP/Triton/PyTorch tasks; complete agents and unseen-configuration generalization | — |

### Cross-Platform Benchmarks

| Year | Work | Focus | Resources |
|---|---|---|---|
| 2025 | [MultiKernelBench](https://arxiv.org/abs/2507.17773) | 285 operators with CUDA, AscendC and Pallas backends in the surveyed release | [Code](https://github.com/wzzll123/MultiKernelBench) |

### Production Benchmarks

| Year | Work | Focus | Resources |
|---|---|---|---|
| 2026 | [FlashInfer-Bench](https://arxiv.org/abs/2601.00227) | Serving-trace workloads, continuous evaluation and deployment-oriented kernel substitution | [Code](https://github.com/flashinfer-ai/flashinfer-bench) |

### Unseen-ISA Adaptation Benchmarks

| Year | Work | Focus | Resources |
|---|---|---|---|
| 2026 | [KernelCraft](https://arxiv.org/abs/2603.08721) | Assembly kernel generation on emerging accelerators using ISA documentation and tool feedback | [Project](https://kernelcraft-cam.github.io/) |

## Kernel Generation and Optimization Techniques

The four categories below follow Section 5.1 and Table 5 of the survey. They contain **30 CUDA-centric methods, 15 high-level DSL methods, 8 alternative-accelerator methods and 9 cross-platform/multi-backend methods**. Works are listed once by their primary focus, even when they support additional backends or combine search, training and agent orchestration.

### 1. CUDA-Centric Kernel Optimization

#### Single-Agent

| Year | Work | Focus | Resources |
|---|---|---|---|
| 2025 | [NVIDIA / DeepSeek-R1 workflow](https://developer.nvidia.com/blog/automating-gpu-kernel-generation-with-deepseek-r1-and-inference-time-scaling/) | Verifier-guided CUDA generation with inference-time scaling | — |
| 2025 | [CUDA-LLM](https://arxiv.org/abs/2506.09092) | Feature Search and Reinforcement with compilation, correctness and latency feedback | — |
| 2025 | [Tutoring LLM into a Better CUDA Optimizer](https://doi.org/10.1007/978-3-031-99857-7_18) | Human optimization hints for CUDA assignments | [Code](https://github.com/matyas-brabec/2025-europar-llm) |
| 2025 | [EvoEngineer](https://arxiv.org/abs/2510.03760) | Evolutionary CUDA code search and population management | — |
| 2026 | [OptiML](https://arxiv.org/abs/2602.12305) | CUDA synthesis followed by profiler-guided Monte Carlo tree search | — |
| 2026 | [K-Search](https://arxiv.org/abs/2602.19128) | Co-evolving world model for optimization-intent search; CUDA and Triton | [Code](https://github.com/caoshiyi/K-Search) |
| 2026 | [CUDA-Harness](https://arxiv.org/abs/2609.00058) | Natural-language-to-CUDA generation, synthesis-based verification and adaptive refinement | — |

#### Multi-Agent

| Year | Work | Focus | Resources |
|---|---|---|---|
| 2025 | [AKG Kernel Agent (AIKG)](https://atomgit.com/mindspore/akg) | Agent orchestration with Triton, SWFT, CUDA C++, C++ and TileLang | [Code](https://atomgit.com/mindspore/akg) |
| 2025 | [How Many Agents Does it Take to Beat PyTorch?](https://letters.lossfunk.com/p/how-many-agents-does-it-take-to-beat) | Orchestrated strategy generation, synthesis, compilation and correctness checking | — |
| 2025 | [AI CUDA Engineer / robust-kbench](https://arxiv.org/abs/2509.14279) | Verification-guided evolutionary CUDA optimization | [Benchmark](https://github.com/SakanaAI/robust-kbench) |
| 2025 | [Astra](https://arxiv.org/abs/2509.07506) | Testing, profiling, planning and coding agents for production CUDA kernels | [Code](https://github.com/Anjiang-Wei/Astra) |
| 2025 | [CudaForge](https://arxiv.org/abs/2511.01884) | Coder–Judge refinement with correctness and hardware feedback | [Code](https://github.com/OptimAI-Lab/CudaForge) |
| 2025 | [The FM Agent](https://arxiv.org/abs/2510.26144) | Parallel generation and island-based evolutionary search | — |
| 2025 | [STARK](https://arxiv.org/abs/2510.16996) | Strategic planning, coding and debugging with search memory | — |
| 2025 | [PIKE](https://arxiv.org/abs/2511.16964) | Population search with mutation, crossover and error fixing | [Code](https://github.com/pike-project/pike) |
| 2026 | [A Two-Stage GPU Kernel Tuner Combining Semantic Refactoring and Search-Based Optimization](https://arxiv.org/abs/2601.12698) | Semantic CUDA refactoring followed by parameter search | — |
| 2026 | [KernelBlaster](https://arxiv.org/abs/2602.14293) | Memory-augmented in-context reinforcement learning for CUDA optimization | — |
| 2026 | [KernelSkill](https://arxiv.org/abs/2603.10085) | Reusable optimization skills with long-term and short-term memory | [Code](https://github.com/0satan0/KernelMem) |
| 2026 | [KEET](https://arxiv.org/abs/2605.04467) | LLM agents that explain GPU profiling evidence | — |
| 2026 | [KernelArc](https://arxiv.org/abs/2608.17071) | Multi-agent exploration across CUDA, PTX, Triton, CuTe and library configurations | — |
| 2026 | [HIERA](https://arxiv.org/abs/2608.21157) | Workload-aware planning across CUDA implementation spaces | — |

#### Training-Based

| Year | Work | Focus | Resources |
|---|---|---|---|
| 2025 | [CUDA-L1](https://arxiv.org/abs/2507.14111) | Contrastive reinforcement learning for CUDA optimization | [Code](https://github.com/ornith-ai/CUDA-L1) |
| 2025 | [Kevin](https://arxiv.org/abs/2507.11948) | Multi-turn reinforcement learning with execution feedback | [Model](https://huggingface.co/cognition-ai/Kevin-32B) |
| 2025 | [ConCuR / KernelCoder](https://arxiv.org/abs/2510.07356) | Training on curated PyTorch–reasoning–CUDA examples | [Model](https://huggingface.co/lkongam/KernelCoder) |
| 2025 | [Integrating Performance Tools in Model Reasoning for GPU Kernel Optimization](https://arxiv.org/abs/2510.17158) | Training models to use performance tools during reasoning | — |
| 2026 | [SparseRL](https://proceedings.iclr.cc/paper_files/paper/2026/file/32400dbdd8b99abe47a49e43dcb1fcf5-Paper-Conference.pdf) | Pretrained models and reinforcement learning for sparse CUDA generation | — |
| 2026 | [CUDA Agent](https://arxiv.org/abs/2602.24286) | Large-scale agentic reinforcement learning for CUDA kernel generation | [Project](https://cuda-agent.github.io/) |
| 2026 | [LEAP](https://arxiv.org/abs/2608.01804) | Adaptive pruning to reduce environment-feedback cost in kernel-generation RL | — |

#### Structured CUDA Abstractions

| Year | Work | Focus | Resources |
|---|---|---|---|
| 2026 | [CuTeGen](https://arxiv.org/abs/2604.01489) | Agentic generation of CuTe-based CUDA C++ kernels | [Code](https://github.com/taratt/cutegen) |
| 2026 | [KernelPro](https://arxiv.org/abs/2606.26453) | Raw CUDA and CuTe generation with micro-profiling tools and structured search | — |

### 2. High-Level Kernel DSL Optimization

#### Single-Agent

| Year | Work | Focus | Resources |
|---|---|---|---|
| 2025 | [KernelBand](https://arxiv.org/abs/2511.18868v1) | Hardware-aware multi-armed bandit search for Triton optimization | — |
| 2026 | [Kernel Foundry (Huang et al.)](https://arxiv.org/abs/2605.30359) | Diagnosis-driven evolutionary Triton optimization with expert guidance | — |
| 2026 | [LLM4LLM](https://arxiv.org/abs/2608.21836) | Closed-loop Triton optimization spanning benchmarks and LLM deployment | — |

#### Multi-Agent

| Year | Work | Focus | Resources |
|---|---|---|---|
| 2025 | [GEAK](https://arxiv.org/abs/2507.23194) | Agentic Triton generation and evaluation for AMD GPUs | [Agent](https://github.com/AMD-AGI/GEAK) · [Evaluation](https://github.com/AMD-AGI/GEAK-eval) |
| 2025 | [KernelFalcon](https://pytorch.org/blog/kernelfalcon-autonomous-gpu-kernel-generation-via-deep-agents/) | Autonomous Triton generation via deep agents | [Code](https://github.com/meta-pytorch/KernelAgent) |
| 2025 | [PRAGMA](https://arxiv.org/abs/2511.06345) | Profiling-reasoned multi-agent kernel optimization | — |
| 2025 | [TritonForge](https://arxiv.org/abs/2512.09196) | Profiling-guided agentic Triton refinement | — |
| 2026 | [KernelAgent](https://pytorch.org/blog/kernelagent-hardware-guided-gpu-kernel-optimization-via-multi-agent-orchestration/) | Hardware-guided profiling, diagnosis and multi-agent Triton optimization | [Code](https://github.com/meta-pytorch/KernelAgent) |
| 2026 | [KernelOPT](https://arxiv.org/abs/2609.30059) | Dispatch-aware agentic search over Triton, Helion and compiled-model rewrites | — |
| 2026 | [AI as a Compiler](https://arxiv.org/abs/2609.36800) | Agentic Triton-to-PTX compilation without the Triton compiler | — |

#### Training-Based

| Year | Work | Focus | Resources |
|---|---|---|---|
| 2025 | [KernelLLM](https://huggingface.co/facebook/KernelLLM) | Supervised fine-tuning for PyTorch-to-Triton generation | [Model](https://huggingface.co/facebook/KernelLLM) · [Dataset](https://huggingface.co/datasets/GPUMODE/KernelBook) |
| 2025 | [AutoTriton](https://arxiv.org/abs/2507.05687) | Supervised fine-tuning and reinforcement learning for Triton synthesis | [Code](https://github.com/AI9Stars/AutoTriton) |
| 2025 | [TritonRL](https://arxiv.org/abs/2510.17891) | Hierarchical rewards and verification against reward hacking | — |
| 2025 | [QiMeng-Kernel (MTMC)](https://arxiv.org/abs/2511.20100) | Macro-level strategy learning and incremental micro-level implementation | [Code](https://github.com/QiMeng-IPRC/QiMeng-Kernel) |
| 2026 | [Dr. Kernel](https://arxiv.org/abs/2602.05885) | Multi-turn RL, profiling-based rewards and the KernelGYM environment | [Code / environment](https://github.com/hkust-nlp/KernelGYM) |

### 3. Optimization for Alternative Accelerators

#### AMD GPUs and ROCm/HIP

| Year | Work | Focus | Resources |
|---|---|---|---|
| 2025 | [GPU Kernel Scientist](https://arxiv.org/abs/2506.20807) | HIP generation through hypothesis, experiment and validation agents | — |
| 2025 | [SwizzlePerf](https://arxiv.org/abs/2508.20258) | Hardware-aware swizzling policies for locality and cache behavior | — |
| 2026 | [ARGUS](https://arxiv.org/abs/2604.18616v1) | Data-flow invariants and static feedback for a tile-based Pythonic DSL | — |
| 2026 | [AsmEvo](https://arxiv.org/abs/2608.20711) | AMDGCN assembly optimization with empirical functional-equivalence checks | — |

#### Ascend and NPU Backends

| Year | Work | Focus | Resources |
|---|---|---|---|
| 2026 | [AscendKernelGen](https://arxiv.org/abs/2601.07160) | AscendC training with Ascend-CoT and NPUKernelBench | [Benchmark](https://github.com/weich97/NPUKernelBench) · [Models](https://huggingface.co/AscendKernelGen) |
| 2026 | [SAGE — From Experience to Expertise](https://arxiv.org/abs/2609.35568) | Adoption-aware experience reuse for data-scarce NPU kernel synthesis | — |

#### Intel GPUs and Other Accelerators

| Year | Work | Focus | Resources |
|---|---|---|---|
| 2026 | [Xe-Forge](https://arxiv.org/abs/2605.26118) | Hardware-aware, multi-stage Triton optimization for Intel GPUs | — |
| 2026 | [MaxKernel](https://arxiv.org/abs/2609.04523) | Multi-agent Pallas kernel generation and optimization for Google TPUs | [Code](https://github.com/AI-Hypercomputer/accelerator-agents/tree/main/MaxKernel) |

### 4. Cross-Platform and Multi-Backend Optimization

#### Cross-Platform Agentic Optimization

| Year | Work | Focus | Resources |
|---|---|---|---|
| 2025 | [KForge](https://arxiv.org/abs/2511.13274) | Cooperating generation and profiling agents targeting CUDA and Apple Metal | — |
| 2025 | [GPU Kernel Optimization Beyond Full Builds](https://arxiv.org/abs/2512.22147) | Minimal executable programs for CUDA/HIP optimization outside full application builds | — |
| 2026 | [Kernel-Smith](https://arxiv.org/abs/2603.28342) | Evolutionary search and post-training across Triton and MACA backends | — |

#### Multi-Backend Optimization

| Year | Work | Focus | Resources |
|---|---|---|---|
| 2026 | [AutoKernel](https://arxiv.org/abs/2603.21331) | Bottleneck prioritization and keep-or-revert refinement for Triton/CUDA | [Code](https://github.com/RightNow-AI/autokernel) |
| 2026 | [KernelFoundry (Wiedemann et al.)](https://arxiv.org/abs/2603.12440) | Quality-diversity search and prompt evolution across kernel backends | — |
| 2026 | [GPU Forecasters](https://arxiv.org/abs/2605.31464) | Selective runtime surrogates to reduce physical GPU evaluations | [Code](https://github.com/codezakh/gpu-forecasters) |
| 2026 | [KLineage](https://arxiv.org/abs/2605.28213v1) | Verified optimization skills recovered from expert kernel lineages | [Code](https://github.com/ict-agent/klineage) |
| 2026 | [daVinci-kernel](https://arxiv.org/abs/2606.16497) | Joint RL for skill selection, kernel optimization and skill summarization | — |
| 2026 | [KernelZero](https://arxiv.org/abs/2609.33074) | Co-evolving task proposer and coder with separate CUDA and Triton models | — |

## Evaluation Dimensions

The survey compares evaluation practices along seven dimensions:

| Dimension | What to assess |
|---|---|
| Functional correctness | Compilation, executable behavior and numerical agreement with reference computations |
| Acceleration performance | Runtime improvement against clearly specified baselines |
| Robustness | Correctness and performance across unseen shapes, seeds, edge cases and execution settings |
| Efficiency | Cost of obtaining a kernel: optimization time, model inference, candidate evaluations, compilation, profiling and GPU time |
| Versatility | Coverage of operators, workload families, tensor shapes and numerical formats |
| Cross-architecture portability | Transfer across accelerator architectures and programming backends |
| Reproducibility | Availability of artifacts and controlled, repeatable correctness/performance protocols |

Reported speedups should be read together with hardware, baseline, input configuration, measurement protocol and optimization budget. Kernel runtime and the cost of discovering that kernel are separate evaluation targets.

## Research Roadmap

Section 6 identifies complementary directions for advancing the field:

- **Broader evaluation:** assess robustness, portability, optimization cost and production integration alongside correctness and speedup.
- **Reusable human expertise:** organize optimization strategies into skills with explicit preconditions and validate their applicability through compilation, testing and measurement.
- **Integration with established optimization:** combine LLM reasoning with auto-tuning, schedule search, compiler transformations, polyhedral analysis and structured kernel abstractions.
- **Kernel-specific code models:** improve execution-semantic understanding, performance-aware generation and hardware-aware cost modeling.

## Related Surveys and Background

These resources provide context and are **not additional members of the 76-study primary corpus**.

| Year | Work | Relevance |
|---|---|---|
| 2023 | [Optimization Techniques for GPU Programming](https://doi.org/10.1145/3570638) | Established GPU optimization techniques and their relationships |
| 2024 | [Large Language Models for Software Engineering: A Systematic Literature Review](https://doi.org/10.1145/3695988) | Broader LLM-assisted software engineering context |
| 2026 | [A Survey on Large Language Models for Code Generation](https://doi.org/10.1145/3747588) | General-purpose code language models and generation methods |
| 2025 | [CuAsmRL: Optimizing GPU SASS Schedules via Deep Reinforcement Learning](https://doi.org/10.1145/3696443.3708943) | Related reinforcement-learning-based instruction scheduling; not an LLM-centered method |

Useful documentation: [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) · [Triton](https://triton-lang.org/main/index.html) · [Pallas](https://docs.jax.dev/en/latest/pallas/index.html) · [Nsight Compute](https://docs.nvidia.com/nsight-compute/)

## Common Terms

| Term | Meaning |
|---|---|
| AOT / JIT | Ahead-of-time / just-in-time compilation |
| CTA | Cooperative thread array |
| CuTe | CUDA C++ abstractions for composing thread/value layouts and tiled computations |
| DSL | Domain-specific language; Triton and Pallas are kernel DSLs |
| FSR | Feature Search and Reinforcement, the framework introduced by CUDA-LLM |
| GEMM | General matrix multiplication |
| HIP | Heterogeneous-compute Interface for Portability |
| ILP | Instruction-level parallelism |
| ISA | Instruction set architecture |
| LLM | Large language model |
| MCTS | Monte Carlo tree search |
| NCU | NVIDIA Nsight Compute, a GPU kernel profiler |
| NPU / TPU | Neural processing unit / tensor processing unit |
| PTX | Parallel Thread Execution, NVIDIA's virtual ISA |
| RL / SFT | Reinforcement learning / supervised fine-tuning |
| ROCm | AMD's GPU software platform |
| SASS | NVIDIA GPU native machine instructions |
| SM | Streaming multiprocessor |
| TMA | Tensor Memory Accelerator |

## Contributing

Contributions and corrections are welcome. Please open an issue or pull request and follow [CONTRIBUTING.md](./CONTRIBUTING.md).

For a new entry, include its title, first public availability date, paper or first-party source, target hardware/backends, benchmark or method category, and available artifacts. Distinguish released implementations from supporting artifacts and announced releases. When one work introduces both a benchmark and an optimizer, link the entries and count it once when reporting independent studies.

## Citation

If this repository or survey is useful for your research, please cite the survey. The entry below describes the current revised manuscript.

```bibtex
@misc{li2026dlkernel,
  title = {Recent Advances in Deep Learning Kernel Optimization Using Large Language Models},
  author = {Li, Tianlin and Yang, Chenxi and Sun, Qitong and Zhang, Xiaoyu and Hu, Qiang and Tang, Zhe and Chen, Sheng and Yang, Fei and Liu, Aishan and Liu, Xianglong and Shen, Chao and Le Traon, Yves and Liu, Yang},
  year = {2026},
  howpublished = {Survey manuscript},
  url = {https://github.com/luckily268/Awesome-GPU-Kernel-Optimization}
}
```

## License

This repository is licensed under the [MIT License](./LICENSE).
