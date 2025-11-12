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
- LLMs for code generation and refinement (e.g., Code Llama, GPT-4)
- Reinforcement learning and feedback-driven kernel synthesis
- Agent-based GPU optimization pipelines
- Multi-modal datasets for performance learning

### 3. Key Concepts
- **Kernel Abstraction & IRs:** Triton IR, TVM IR, MLIR, KPerfIR  
- **Optimization Objectives:** throughput, latency, occupancy, register usage  
- **Evaluation Metrics:** performance gain, energy efficiency, generalization  

### 4. Benchmark Suites & Datasets
- MLPerf, DeepBench, CUTLASS benchmarks  
- Synthetic kernel datasets for training & evaluation  
- GPU operator profiling repositories  

---

## 🧠 Representative Frameworks

| Framework | Type | Era | Description |
|------------|------|------|-------------|
| **TVM / AutoTVM** | Compiler-based | Pre-LLM | Template-driven kernel optimization and auto-tuning |
| **TACO** | Compiler | Pre-LLM | Sparse tensor algebra compiler |
| **Triton** | DSL | Transition | Python-based GPU kernel programming and code generation |
| **GEAK** | AI-driven | Post-LLM | End-to-end GPU kernel optimization via LLMs and feedback |
| **Mirage / DeepTune** | Model-based | Post-LLM | Neural kernel synthesis and performance prediction |

---

## 🧪 Research Topics Covered

- GPU operation and memory hierarchy
- Automatic kernel synthesis and tuning
- Program representation learning
- Feedback-based and self-evolving optimizers
- Evaluation and benchmarking methodology

---

## 🔍 Related Surveys

| Title | Year | Focus |
|-------|------|--------|
| “Auto-tuning of GPU Kernels: Techniques and Trends” | 2020 | Pre-LLM tuning methodologies |
| “From Manual Tuning to AI-driven Optimization” | 2025 | LLM-assisted GPU kernel optimization |
| “Learning to Optimize Compilers” | 2023 | ML for compiler decision-making |
