# KokoroSystem ARC-AGI-2: Solar + Spider Stable

**Achieved 64.17% accuracy on ARC-AGI-2 evaluation set (120 tasks) in ~14.46s.**  
This implementation combines two generative transformation frameworks from the KokoroSystem project, developed by **Yuki Hoshino** (Japan):

1. **Solar System Model**  
   - Treats the entire input pattern as the "Sun" (core structure).
   - Generates "planets" (rotations, flips, color mappings) and "satellites" (compound transformations).
   - Efficiently covers whole-structure transformation tasks.

2. **Spider Abstract Method**  
   - Treats partial patterns (difference regions) as central nodes.
   - Generates first-layer variants (basic transformations) and second-layer combinations.
   - Covers partial replacement and reconfiguration tasks.

---

## Results

| Metric            | Value   |
|-------------------|---------|
| Tasks evaluated   | 120     |
| Correct           | 77      |
| Accuracy          | **64.17%** |
| Processing time   | ~14.46s |

**Position:**  
This score is among the top-tier results for ARC-AGI-2 with a *pure rule-based general algorithm*, without task-specific tuning. Achieved independently by Yuki Hoshino.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16756179.svg)](https://doi.org/10.5281/zenodo.16756179)

---

## Requirements

- Python 3.9+
- numpy
- opencv-python

Install dependencies:
```bash
pip install numpy opencv-python
