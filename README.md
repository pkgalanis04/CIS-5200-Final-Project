# Spaced Repetition Optimization System

A machine learning and reinforcement learning system to optimize spaced repetition scheduling for language learning, using Duolingo review data.

## Project Overview

This project compares four different approaches to spaced repetition scheduling:

1. **SM-2 Baseline**: Traditional SuperMemo 2 heuristic algorithm
2. **Logistic Regression**: Supervised learning to predict recall probability
3. **Multi-Armed Bandit RL**: Context-free RL grouping items by difficulty
4. **DQN (Deep Q-Network)**: Full model-free RL with neural network function approximation

## Project Structure

```
/project
    /data                    # Processed data files
    /environment
        memory_env.py        # RL environment for spaced repetition
    /models
        sm2_baseline.py      # SM-2 algorithm implementation
        logistic_model.py    # Logistic regression model
        bandit_agent.py      # Multi-armed bandit agent
        dqn_agent.py         # DQN agent
    /utils
        preprocess.py        # Data preprocessing
        evaluation.py        # Evaluation metrics and comparison
    main.py                  # Main execution script
    requirements.txt         # Python dependencies
    README.md               # This file
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. (Optional) For GPU support with DQN:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### Step 1: Preprocess the Dataset

First, download and preprocess the Duolingo dataset:

```bash
python main.py --preprocess
```

This will:
- Download the dataset from Kaggle using `kagglehub`
- Extract relevant columns
- Create features (time since last review, historical accuracy, etc.)
- Normalize and encode data
- Split into train/validation/test sets
- Save to `data/processed.pkl`

Alternatively, if you already have the dataset:
```bash
python main.py --preprocess --dataset_path /path/to/dataset
```

### Step 2: Train and Evaluate Models

Train all models:
```bash
python main.py --model all --episodes 50
```

Train a specific model:
```bash
python main.py --model dqn --episodes 100
python main.py --model sm2 --episodes 20
python main.py --model logistic --episodes 30
python main.py --model bandit --episodes 50
```

### Command Line Arguments

**Data:**
- `--preprocess`: Preprocess the dataset
- `--dataset_path`: Path to dataset (if not using kagglehub)

**Model Selection:**
- `--model`: Which model(s) to train (`all`, `sm2`, `logistic`, `bandit`, `dqn`)

**Training:**
- `--episodes`: Number of training episodes (default: 50)
- `--time_budget`: Time budget for reviews (default: 100.0)
- `--forgetting_rate`: Memory decay rate (default: 0.1)
- `--seed`: Random seed (default: 42)

**Bandit:**
- `--num_arms`: Number of difficulty groups (default: 10)
- `--epsilon`: Exploration rate (default: 0.1)

**DQN:**
- `--learning_rate`: Learning rate (default: 0.001)
- `--batch_size`: Mini-batch size (default: 32)
- `--use_gpu`: Use GPU for training

**Output:**
- `--save_models`: Save trained model checkpoints
- `--output_dir`: Output directory for results (default: `results`)

### Example Commands

```bash
# Quick test with fewer episodes
python main.py --model all --episodes 10

# Full training with GPU
python main.py --model dqn --episodes 200 --use_gpu --save_models

# Compare SM-2 and Logistic Regression
python main.py --model sm2 --episodes 30
python main.py --model logistic --episodes 30
```

## Evaluation Metrics

The system computes and compares:

1. **Recall Rate**: Percentage of correct recalls (higher is better)
2. **Intervention Efficiency**: Average time between correct recalls (lower is better)
3. **Cumulative Reward**: Sum of rewards over episodes (higher is better)

Results are saved to:
- `results/model_comparison.csv`: Comparison table
- `results/training_curves_*.png`: Training curves
- `results/comparison_*.png`: Bar charts comparing models

## Model Details

### SM-2 Baseline
- Implements SuperMemo 2 algorithm
- Uses ease factor (EF) and intervals
- Selects items with earliest next review time

### Logistic Regression
- Supervised learning on historical review data
- Predicts recall probability from features
- Selects items with highest predicted probability

### Multi-Armed Bandit
- Groups items by difficulty (historical accuracy)
- Uses epsilon-greedy policy
- Incremental Q-value updates

### DQN (Deep Q-Network)
- Neural network Q-function approximation
- Experience replay buffer
- Target network for stable training
- Epsilon-greedy exploration with decay

## Hyperparameters

### Recommended Settings

**SM-2:**
- Initial EF: 2.5
- Min EF: 1.3

**Logistic Regression:**
- C: 1.0 (regularization)
- Max iterations: 1000

**Bandit:**
- Num arms: 10
- Epsilon: 0.1
- Learning rate: 0.1

**DQN:**
- Learning rate: 0.001
- Batch size: 32
- Epsilon start: 1.0, end: 0.01, decay: 0.995
- Gamma (discount): 0.99
- Target update frequency: 10
- Replay buffer capacity: 10000

## Output Files

After training, you'll find:

- `data/processed.pkl`: Preprocessed dataset
- `results/model_comparison.csv`: Comparison metrics
- `results/training_curves_recall_rate.png`: Recall rate over episodes
- `results/training_curves_cumulative_reward.png`: Reward over episodes
- `results/comparison_recall_rate.png`: Bar chart comparison
- `results/comparison_cumulative_reward.png`: Reward comparison
- `checkpoints/*.pkl` or `checkpoints/*.pt`: Saved models (if `--save_models`)

## Troubleshooting

**Dataset not found:**
- Make sure to run `--preprocess` first
- Check that `kagglehub` is installed: `pip install kagglehub`

**Out of memory (DQN):**
- Reduce `--batch_size` (e.g., `--batch_size 16`)
- Reduce number of episodes
- Use CPU instead of GPU: remove `--use_gpu`

**Slow training:**
- Reduce `--episodes`
- Use smaller dataset subset
- For DQN, reduce `--batch_size` or use fewer items

## Future Improvements

- Add more sophisticated state representations
- Implement attention mechanisms for DQN
- Add curriculum learning
- Implement actor-critic methods (A3C, PPO)
- Add user-specific personalization
- Implement transfer learning across users

## License

This project is for educational/research purposes.

## Citation

If using the Duolingo dataset, please cite:
```
@dataset{aravinii_duolingo,
  title={Duolingo Spaced Repetition Data},
  author={Aravinii},
  year={2024},
  url={https://www.kaggle.com/datasets/aravinii/duolingo-spaced-repetition-data}
}
```

