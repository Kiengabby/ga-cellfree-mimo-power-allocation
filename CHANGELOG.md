# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-18

### Added
- Initial release of Genetic Algorithm for Power Allocation in Cell-Free Massive MIMO
- **Standard GA Implementation**:
  - Real-coded genetic algorithm with Tournament Selection (k=3)
  - Arithmetic Crossover for parent combination
  - Gaussian Mutation for maintaining diversity
  - Repair mechanism for constraint handling
  - Elitism strategy (10% preservation)
  
- **Adaptive GA Variant**:
  - Time-varying mutation rate (pm: 0.5 → 0.01)
  - Enhanced elitism mechanism
  - Fine-tuning with decreasing noise scale
  - Tournament-3 selection for higher selection pressure
  
- **System Model**:
  - Cell-Free Massive MIMO with M=10 APs and K=5 UEs
  - Large-scale fading model (path loss + shadow fading)
  - SINR-based Sum-Rate calculation
  - Power constraint handling (P_max = 100 mW per AP)
  
- **Visualization Tools**:
  - Convergence plot (best/average fitness over generations)
  - Power allocation heatmap (AP × UE matrix)
  - Performance comparison (GA vs Baseline)
  - Variant comparison (Standard GA vs Adaptive GA)
  
- **Documentation**:
  - Comprehensive README with badges and examples
  - Detailed Vietnamese comments in all source files
  - LaTeX report with theoretical background
  - Presentation guide for academic defense
  - Contributing guidelines
  - Citation file (CITATION.cff)
  
- **Code Organization**:
  - Modular structure: src/, results/, docs/
  - Three implementations: full, simple, and comparison
  - Professional .gitignore for Python projects
  - Requirements.txt with minimal dependencies
  - MIT License for open-source distribution

### Performance
- **Standard GA**: ~3.05 bits/s/Hz (+89.5% vs Baseline)
- **Adaptive GA**: ~3.50 bits/s/Hz (+14.9% vs Standard GA)
- **Baseline (Equal Power)**: ~1.61 bits/s/Hz

### Dependencies
- Python >= 3.8
- NumPy >= 1.21.0
- Matplotlib >= 3.4.0

---

## Future Work

Planned improvements for future releases:

### [1.1.0] - Planned
- [ ] Add multi-objective optimization (Sum-Rate + Energy Efficiency)
- [ ] Implement NSGA-II for Pareto front generation
- [ ] Add sensitivity analysis tools
- [ ] Include statistical significance tests

### [1.2.0] - Planned
- [ ] Add support for different AP/UE configurations
- [ ] Implement Island Model GA for parallelization
- [ ] Add benchmark against other optimization methods
- [ ] Create interactive visualization dashboard

### [2.0.0] - Planned
- [ ] Extend to dynamic scenarios (user mobility)
- [ ] Add imperfect CSI modeling
- [ ] Implement deep learning hybrid approaches
- [ ] Add real-world dataset integration

---

## Release Notes

### Version 1.0.0 Highlights

This initial release provides a complete, production-ready implementation of Genetic Algorithm for power allocation optimization in Cell-Free Massive MIMO systems. The code is:

✅ **Well-documented**: Comprehensive Vietnamese comments for educational purposes  
✅ **Modular**: Clean separation of concerns (system model, GA engine, visualization)  
✅ **Tested**: Verified results matching theoretical expectations  
✅ **Reproducible**: Fixed random seed for consistent results  
✅ **Professional**: Follows Python best practices and academic standards

The repository is suitable for:
- 🎓 Academic coursework and research projects
- 📚 Learning genetic algorithms and wireless optimization
- 🔬 Baseline for comparing new optimization methods
- 🏗️ Foundation for extended research directions

---

*For detailed commit history, see [GitHub Commits](https://github.com/Kiengabby/ga-cellfree-mimo-power-allocation/commits)*
