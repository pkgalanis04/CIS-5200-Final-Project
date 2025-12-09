import numpy as np

class ToySRSFinalExam:
    def __init__(
        self,
        n_items=10,
        study_len=40,          # length of study phase
        seed=0,
        time_cost=0.01,        # cost per review (lambda)
        exam_bonus_scale=3.0,  # weight on exam performance
    ):

        self.n_items = n_items
        self.study_len = study_len
        self.time_cost = time_cost
        self.exam_bonus_scale = exam_bonus_scale

        self.rng = np.random.default_rng(seed)

        # item parameters
        self.difficulty = self.rng.uniform(0.5, 2.0, size=n_items)
        self.halflife = self.rng.uniform(2.0, 6.0, size=n_items)

        # Discretization for state representation
        self.lag_bins = np.array([1, 5, 10, 20, 50, 100], dtype=float)
        self.strength_bins = np.array([0.5, 1.0, 1.5, 2.0, 3.0], dtype=float)

        self.reset()

    def reset(self):
        # Time since last review for each item
        self.lags = np.zeros(self.n_items, dtype=float)
        # Memory strength for each item
        self.strength = np.ones(self.n_items, dtype=float)
        # Internal time step
        self.t = 0

        self.total_correct = 0.0
        self.total_exam_correct = 0.0

        return self._get_remaining_steps()

    def _get_remaining_steps(self):
        return self.study_len - self.t

    def _recall_prob(self, i):
        # p_i = exp( - lag_i / (halflife_i * strength_i) )
        return np.exp(- self.lags[i] / (self.halflife[i] * self.strength[i]))

    def _discretize(self, values, bins):
        # np.digitize returns indices 0..len(bins)
        return np.digitize(values, bins=bins, right=True)

    def _get_item_features(self, item_idx):
        lag_bin = self._discretize(
            np.array([self.lags[item_idx]]), self.lag_bins
        )[0]
        str_bin = self._discretize(
            np.array([self.strength[item_idx]]), self.strength_bins
        )[0]
        remaining = self._get_remaining_steps()
        remaining_bin = min(remaining // 5, 5)
        return (item_idx, lag_bin, str_bin, remaining_bin)

# Checks what the state of the item will be if they review the item
    def get_state_for_item(self, item_idx):
        return self._get_item_features(item_idx)

    def step(self, action):
        # 1. Advance time: all lags increase by 1
        self.lags += 1
        self.t += 1

        item = action

        # 2. Simulate recall outcome for the chosen item
        p = self._recall_prob(item)
        correct = float(self.rng.random() < p)

        # 3. Update strength based on outcome
        if correct:
            # strengthen more for difficult items
            self.strength[item] *= (1.0 + 0.1 / self.difficulty[item])
        else:
            self.strength[item] *= (1.0 - 0.1 * self.difficulty[item])
            self.strength[item] = max(self.strength[item], 0.1)

        # Reset lag for the reviewed item
        self.lags[item] = 0.0

        # Bookkeeping
        self.total_correct += correct

        # 4. Immediate reward: correctness minus time cost
        reward = correct - self.time_cost

        done = False
        exam_reward = 0.0

        # 5. If we finished the study phase, do a final exam
        if self.t >= self.study_len:
            done = True

            # Recall probabilities at exam time for each item
            exam_probs = np.exp(
                - self.lags / (self.halflife * self.strength)
            )
            exam_outcomes = (self.rng.random(self.n_items) < exam_probs).astype(float)
            exam_correct = exam_outcomes.sum()
            self.total_exam_correct = float(exam_correct)

            exam_reward = self.exam_bonus_scale * exam_correct
            reward += exam_reward  # terminal bonus

        next_state = self._get_item_features(item)
        info = {
            "immediate_correct": float(correct),
            "exam_reward": float(exam_reward),
            "total_exam_correct": float(self.total_exam_correct),
        }
        return next_state, float(reward), done, info

from collections import defaultdict
import numpy as np

env_seed = 0

def make_env():
    return ToySRSFinalExam(
        n_items=50,
        study_len=40,
        time_cost=0.005,
        exam_bonus_scale=5.0,
        seed=env_seed
    )

def evaluate_policy_stateful(env_maker, policy_fn, n_episodes=200):
    rewards = []
    exam_scores = []

    for ep in range(n_episodes):
        env = env_maker()
        env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = policy_fn(env)
            _, r, done, info = env.step(action)
            total_reward += r

        rewards.append(total_reward)
        exam_scores.append(info["total_exam_correct"])

    return float(np.mean(rewards)), float(np.mean(exam_scores))

def train_q_learning(
    n_episodes=5000,
    alpha=0.1,
    gamma=0.95,
    epsilon=0.1
):
    Q = defaultdict(float)

    def epsilon_greedy(env):
        # choose action based on current env state (lags, strengths, time)
        if np.random.rand() < epsilon:
            return np.random.randint(env.n_items)
        qs = []
        for a in range(env.n_items):
            s_a = env.get_state_for_item(a)
            qs.append(Q[(s_a, a)])
        return int(np.argmax(qs))

    for ep in range(n_episodes):
        env = make_env()
        env.reset()
        done = False

        while not done:
            # choose action based on current state
            action = epsilon_greedy(env)
            state = env.get_state_for_item(action)

            _, reward, done, info = env.step(action)

            # compute best next-state value
            if not done:
                best_next = max(
                    Q[(env.get_state_for_item(a), a)]
                    for a in range(env.n_items)
                )
            else:
                best_next = 0.0

            Q[(state, action)] = (
                (1 - alpha) * Q[(state, action)]
                + alpha * (reward + gamma * best_next)
            )

        if (ep + 1) % 1000 == 0:
            print(f"[Q-learning] Episode {ep+1}/{n_episodes}")

    return Q

def make_q_policy(Q):
    def policy(env):
        qs = []
        for a in range(env.n_items):
            s_a = env.get_state_for_item(a)
            qs.append(Q[(s_a, a)])
        return int(np.argmax(qs))
    return policy

# Train Q-learning
Q = train_q_learning(n_episodes=5000)
q_policy = make_q_policy(Q)

def train_eps_bandit(
    n_episodes=5000,
    epsilon=0.1
):
    n_items = make_env().n_items
    est = np.zeros(n_items)
    counts = np.zeros(n_items)

    for ep in range(n_episodes):
        env = make_env()
        env.reset()
        done = False

        while not done:
            if np.random.rand() < epsilon:
                action = np.random.randint(n_items)
            else:
                action = int(np.argmax(est))

            _, reward, done, info = env.step(action)

            counts[action] += 1
            est[action] += (reward - est[action]) / counts[action]

        if (ep + 1) % 1000 == 0:
            print(f"[ε-bandit] Episode {ep+1}/{n_episodes}")

    return est

est_eps = train_eps_bandit()

def eps_bandit_policy(env):
    return int(np.argmax(est_eps))

def train_ucb(
    n_episodes=5000,
    c=2.0
):
    n_items = make_env().n_items
    est = np.zeros(n_items)
    counts = np.zeros(n_items) + 1e-9

    for ep in range(n_episodes):
        env = make_env()
        env.reset()
        done = False
        t = 1

        while not done:
            ucb_values = est + c * np.sqrt(np.log(t + 1) / counts)
            action = int(np.argmax(ucb_values))

            _, reward, done, info = env.step(action)

            counts[action] += 1
            est[action] += (reward - est[action]) / counts[action]

            t += 1

        if (ep + 1) % 1000 == 0:
            print(f"[UCB] Episode {ep+1}/{n_episodes}")

    return est

est_ucb = train_ucb()

def ucb_policy(env):
    return int(np.argmax(est_ucb))

q_avg_reward, q_avg_exam = evaluate_policy_stateful(make_env, q_policy, n_episodes=200)
print(f"Q-learning: avg total reward = {q_avg_reward:.2f}, avg exam correct = {q_avg_exam:.2f}")

eps_avg_reward, eps_avg_exam = evaluate_policy_stateful(make_env, eps_bandit_policy, n_episodes=200)
print(f"ε-bandit:   avg total reward = {eps_avg_reward:.2f}, avg exam correct = {eps_avg_exam:.2f}")

ucb_avg_reward, ucb_avg_exam = evaluate_policy_stateful(make_env, ucb_policy, n_episodes=200)
print(f"UCB:        avg total reward = {ucb_avg_reward:.2f}, avg exam correct = {ucb_avg_exam:.2f}")

import matplotlib.pyplot as plt

methods = ['Q-learning', 'ε-bandit', 'UCB']
exam_scores = [q_avg_exam, eps_avg_exam, ucb_avg_exam]
total_rewards = [q_avg_reward, eps_avg_reward, ucb_avg_reward]

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].bar(methods, exam_scores, color=['#4c72b0','#dd8452','#55a868'])
axs[0].set_title("Average Exam Correct")
axs[0].set_ylabel("Items Recalled")
axs[0].set_ylim(0, max(exam_scores)+1)

axs[1].bar(methods, total_rewards, color=['#4c72b0','#dd8452','#55a868'])
axs[1].set_title("Average Total Reward")
axs[1].set_ylabel("Reward")
axs[1].set_ylim(min(total_rewards)-5, max(total_rewards)+5)

plt.tight_layout()
plt.show()

def train_q_learning(
    n_episodes=5000,
    alpha=0.1,
    gamma=0.95,
    epsilon=0.1
):
    Q = defaultdict(float)
    exam_log = []
    reward_log = []

    def epsilon_greedy(env):
        if np.random.rand() < epsilon:
            return np.random.randint(env.n_items)
        qs = [Q[(env.get_state_for_item(a), a)] for a in range(env.n_items)]
        return int(np.argmax(qs))

    for ep in range(n_episodes):
        env = make_env()
        env.reset()
        done = False
        ep_reward = 0

        while not done:
            action = epsilon_greedy(env)
            state = env.get_state_for_item(action)

            _, reward, done, info = env.step(action)
            ep_reward += reward

            if not done:
                best_next = max(Q[(env.get_state_for_item(a), a)] for a in range(env.n_items))
            else:
                best_next = 0.0

            Q[(state, action)] = (
                (1 - alpha) * Q[(state, action)]
                + alpha * (reward + gamma * best_next)
            )

        exam_log.append(info["total_exam_correct"])
        reward_log.append(ep_reward)

    return Q, exam_log, reward_log

Q, exam_log, reward_log = train_q_learning(n_episodes=5000)

plt.figure(figsize=(10,5))
plt.plot(exam_log, label="Exam Correct per Episode", alpha=0.8)
plt.title("Q-learning Exam Performance Over Training")
plt.xlabel("Episode")
plt.ylabel("Items Recalled")
plt.grid(True, alpha=0.4)
plt.show()

plt.figure(figsize=(10,5))
plt.plot(reward_log, label="Total Reward per Episode", color='green', alpha=0.8)
plt.title("Q-learning Total Reward Over Training")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True, alpha=0.4)
plt.show()

