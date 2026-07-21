# Installation

## Requirements

CADRE requires Python **3.10+**.

## Installing from PyPI / Source

Install `cadre` using standard Python package managers (`pip`, `uv`, `poetry`):

```bash
pip install cadre
```

If installing from source:

```bash
git clone https://github.com/your-org/cadre.git
cd cadre
pip install .
```

---

## Key Dependencies

CADRE depends on the following core Python packages:

| Dependency | Purpose |
| :--- | :--- |
| `pydantic >= 2.0.0` | Strict data validation and configuration schemas |
| `numpy >= 1.24.0` | Numerical matrix operations and feature scaling |
| `scipy >= 1.10.0` | Numerical optimization (`minimize`) and statistical functions (`expit`, `beta`) |
| `scikit-learn >= 1.2.0` | Isotonic regression calibration (`IsotonicRegression`) |
| `networkx >= 3.0` | Graph structures for `EvidenceGraph` and `ClaimGraph` |

---

## Verifying Installation

Verify that CADRE is correctly installed in your Python environment:

```python
import cadre

print(cadre.__all__)
```

Output should include core exports such as `CadreEngine`, `CadreConfig`, `TrustedContext`, `MonotoneRiskHead`, and `RuntimeBudget`.
