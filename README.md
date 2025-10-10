# AR-GSE

An Implementation of AR-GSE: Gating Mechanisms in Ensemble Models  for Robust Rejection Learning on  Long-Tail Data


# How to run step-by-step

1. Clone the repository:

```bash
git clone https://github.com/dngback/AR-GSE.git
cd AR-GSE
```

2. Install the required packages:

Recommended to use a virtual environment like `venv` or `conda`. 

```bash
pip install -r requirements.txt
```

3. Run Data Preparation:

```bash
python -c "from src.data.enhanced_datasets import create_full_cifar100_lt_splits; create_full_cifar100_lt_splits()"
```

3. Train expert models: 

```bash
python -m src.train.train_expert
```

4. Train gating model warm up:

```bash
python -m src.train.train_gating_only --mode pretrain
```

5. Train gating for selected expert:

```bash
python -m src.train.train_gating_only --mode selective
```

6. Train AR-GSE:

```bash
python run_improved_eg_outer.py
```

7. Evaluate AR-GSE:

```bash
python -m src.train.eval_agse_plugin.py
```

# Visualization

### 🎯 Demo Inference (Quick Analysis)
For quick demonstration and understanding of a few samples:

```powershell
python demo_inference.py
```

### 🚀 Comprehensive Inference (Research-Grade Analysis)
For comprehensive analysis with 50 samples (30 Head + 20 Tail):

```powershell
python comprehensive_inference.py
```