# CSC 8810 Computational Intelligence Project: Complete Research Roadmap

## Executive Summary
This comprehensive research plan provides a structured, reproducible framework for implementing Deep Fuzzy Rough Graph Neural Networks (DFRG-NNs) with Adaptive Multiobjective Optimization—a novel architecture integrating three core course topics: fuzzy logic, neural networks, and genetic algorithms.

## Section 1: Project Architecture and Scope
### Core Innovation
- **Fuzzy Logic**: Fuzzification layers, fuzzy inference rules, defuzzification.
- **Neural Networks**: Graph convolution blocks, deep stacking.
- **Genetic Algorithms**: NSGA-II for multiobjective optimization.

### Architectural Layers
1. **Input Fuzzification Layer**: Convert crisp features to fuzzy membership values.
2. **Fuzzy Rough Processing Blocks**: Rough set approximators + Fuzzy GCN.
3. **Output Defuzzification**: Center-of-gravity method.
4. **Multiobjective Evolution**: NSGA-II optimization.

## Section 2: Data Sources
- **Primary**: Cora Citation Network (2,708 nodes, 5,429 edges, 7 classes).
- **Validation**: Citeseer.
- **Synthetic**: 9 uncertainty scenarios (Edge Uncertainty, Node Noise, Edge Dropout).

## Section 3: Ablation Study Framework
8 Targeted Experiments:
- **AB-1**: Fuzzy Parameters
- **AB-2**: Rough Set Integration
- **AB-3**: Deep Architecture
- **AB-4**: Multiobjective Opt.
- **AB-5**: Fuzzy Depth
- **AB-6**: Rough Set Type
- **AB-7**: Edge Uncertainty
- **AB-8**: Rule Extraction

## Section 4: Evaluation Metrics
- **Accuracy**: Accuracy, Macro-F1, Micro-F1, ROC-AUC.
- **Interpretability**: # Rules, Rule Simplicity, Completeness.
- **Generalization**: Train-Test Gap, Cross-Val Error.
- **Robustness**: Edge Uncertainty Performance.

## Section 5: Implementation Plan
- **Module 1**: `src/fuzzy_layers.py` (Fuzzification, FuzzyGCN)
- **Module 2**: `src/rough_sets.py` (Approximation, Uncertainty)
- **Module 3**: `src/optimizer.py` (NSGA-II)
- **Module 4**: `experiments/ablation.py` (Study runner)
- **Module 5**: `experiments/metrics.py` (KPIs)
- **Module 6**: `experiments/train.py` (Training loop)

## Section 6: Success Criteria
- Accuracy >= 85% on Cora.
- Interpretable fuzzy rules (<50).
- Robustness to uncertainty.

## Section 7: Timeline
- **Week 1-2**: Design (Done)
- **Week 3-4**: Implementation (Current)
- **Week 5**: Baseline Experiments (Done)
- **Week 6**: Ablation Studies (Done - Found Optimal Shallow Arch 71.7%)
- **Week 6.5**: Architecture Refinement (Done - Hybrid Pre-compression)
- **Week 7**: Multiobjective Optimization (NSGA-II) (Done - Found 20 Pareto Solutions)
- **Week 8**: Documentation
