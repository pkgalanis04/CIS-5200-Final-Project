# Run Instructions - Spaced Repetition Optimization System

## Prerequisites

First, install all required dependencies:
```bash
pip install -r requirements.txt
```

This installs:
- numpy
- pandas
- scikit-learn
- torch (PyTorch)
- matplotlib
- kagglehub

## Step-by-Step Execution

### Step 1: Preprocess the Dataset (REQUIRED - Run First!)

Download and preprocess the Duolingo dataset:

```bash
python main.py --preprocess
```

This will:
- Automatically download the dataset from Kaggle
- Process and clean the data
- Create features (time since last review, historical accuracy, etc.)
- Split into train/validation/test sets
- Save to `data/processed.pkl`

**Expected Output:**
```
PREPROCESSING DUOLINGO DATASET
============================================================
Loading data from: ...
Creating features...
Encoding IDs...
Normalizing features...
Splitting data...
Train: X reviews from Y users
Val: X reviews from Y users
Test: X reviews from Y users
Saving processed data...
Saved to data/processed.pkl
PREPROCESSING COMPLETE
```

### Step 2: Train Models

#### Option A: Train All Models

Train all four models (SM-2, Logistic Regression, Bandit, DQN):

```bash
python main.py --model all --episodes 50
```

#### Option B: Train Individual Models

**SM-2 Baseline (fastest):**
```bash
python main.py --model sm2 --episodes 20
```

**Logistic Regression:**
```bash
python main.py --model logistic --episodes 30
```

**Multi-Armed Bandit RL:**
```bash
python main.py --model bandit --episodes 50
```

**DQN (Deep Q-Network) - Most powerful but slowest:**
```bash
python main.py --model dqn --episodes 100 --use_gpu
```

Or on CPU:
```bash
python main.py --model dqn --episodes 100
```

### Step 3: View Results

After training completes, results are saved to:
- `results/model_comparison.csv` - Comparison table with all metrics
- `results/training_curves_recall_rate.png` - Recall rate over episodes
- `results/training_curves_cumulative_reward.png` - Reward over episodes
- `results/comparison_recall_rate.png` - Bar chart comparison
- `results/comparison_cumulative_reward.png` - Reward comparison

## Quick Start Example

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Preprocess dataset (one time, takes a few minutes)
python main.py --preprocess

# 3. Quick test with all models (few episodes for speed)
python main.py --model all --episodes 10

# 4. Full training with all models
python main.py --model all --episodes 50 --save_models

# 5. Train only the best models with more episodes
python main.py --model dqn --episodes 200 --use_gpu --save_models
```

## Command Line Options

### Basic Options
- `--model`: Which model(s) to train
  - `all` - Train all models
  - `sm2` - SM-2 baseline only
  - `logistic` - Logistic regression only
  - `bandit` - Bandit RL only
  - `dqn` - DQN only

- `--episodes`: Number of training episodes (default: 50)
  - More episodes = better performance but slower
  - Recommended: 20-50 for quick tests, 100-200 for final training

- `--preprocess`: Preprocess the dataset (must run first)

### Advanced Options

**Environment Settings:**
- `--time_budget 100.0` - Time available for reviews (default: 100.0)
- `--forgetting_rate 0.1` - Memory decay rate (default: 0.1)
- `--seed 42` - Random seed for reproducibility (default: 42)

**Bandit Settings:**
- `--num_arms 10` - Number of difficulty groups (default: 10)
- `--epsilon 0.1` - Exploration rate (default: 0.1)

**DQN Settings:**
- `--learning_rate 0.001` - Learning rate (default: 0.001)
- `--batch_size 32` - Mini-batch size (default: 32)
- `--use_gpu` - Use GPU for training (requires CUDA)

**Output Settings:**
- `--save_models` - Save trained models to `checkpoints/` directory
- `--output_dir results` - Output directory for results (default: results)

## Complete Example Commands

### Minimal Run (Quick Test)
```bash
python main.py --preprocess
python main.py --model all --episodes 10
```

### Standard Run (Recommended)
```bash
python main.py --preprocess
python main.py --model all --episodes 50 --save_models
```

### Full Training Run
```bash
python main.py --preprocess
python main.py --model sm2 --episodes 30
python main.py --model logistic --episodes 50
python main.py --model bandit --episodes 100
python main.py --model dqn --episodes 200 --use_gpu --save_models
```

### Research Run (Compare Models)
```bash
# Train each model separately with same settings
python main.py --model sm2 --episodes 100 --seed 42
python main.py --model logistic --episodes 100 --seed 42
python main.py --model bandit --episodes 100 --seed 42 --epsilon 0.1
python main.py --model dqn --episodes 100 --seed 42 --learning_rate 0.001
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"
```bash
pip install torch
# Or for CPU-only:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### "Processed data not found"
You need to preprocess first:
```bash
python main.py --preprocess
```

### "kagglehub not found"
```bash
pip install kagglehub
```

### Out of Memory (DQN)
Reduce batch size or use CPU:
```bash
python main.py --model dqn --episodes 50 --batch_size 16
# Or remove --use_gpu to use CPU
python main.py --model dqn --episodes 50
```

### Slow Training
Start with fewer episodes:
```bash
python main.py --model all --episodes 10
```

## Expected Runtime

- **Preprocessing**: 2-5 minutes (one time)
- **SM-2**: ~1-2 minutes per 20 episodes
- **Logistic Regression**: ~2-5 minutes per 30 episodes
- **Bandit**: ~5-10 minutes per 50 episodes
- **DQN (CPU)**: ~10-20 minutes per 50 episodes
- **DQN (GPU)**: ~5-10 minutes per 50 episodes

## Output Files Structure

After running, you'll have:

```
project/
├── data/
│   └── processed.pkl          # Processed dataset
├── results/
│   ├── model_comparison.csv   # Comparison table
│   ├── training_curves_recall_rate.png
│   ├── training_curves_cumulative_reward.png
│   ├── comparison_recall_rate.png
│   └── comparison_cumulative_reward.png
└── checkpoints/               # If --save_models used
    ├── logistic_model.pkl
    └── dqn_model.pt
```

## Next Steps

1. Run preprocessing: `python main.py --preprocess`
2. Start with quick test: `python main.py --model all --episodes 10`
3. Full training: `python main.py --model all --episodes 50 --save_models`
4. Analyze results in `results/` directory
5. Experiment with hyperparameters to improve performance

