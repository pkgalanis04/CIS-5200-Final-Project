import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


class MemoryEnvReal:
    """
    REAL dataset-driven RL environment for spaced repetition.
    The agent's action selects which item to review; recall is simulated
    using item difficulty, time since last review, and prior successes.
    """

    def __init__(
        self,
        item_features: pd.DataFrame,
        user_traces: Dict[int, List[Dict]],
        time_budget: float = 500.0,
        forgetting_rate: float = 0.1,
        seed: Optional[int] = None
    ):
        self.item_features = item_features.set_index("item_id")
        self.user_traces = user_traces
        self.time_budget = time_budget
        self.forgetting_rate = forgetting_rate

        self.item_ids = list(self.item_features.index)
        self.item_to_idx = {item_id: i for i, item_id in enumerate(self.item_ids)}
        self.num_items = len(self.item_ids)

        if seed is not None:
            np.random.seed(seed)

        self.current_time: Optional[pd.Timestamp] = None
        self.reset()

    def reset(self, user_id: Optional[int] = None) -> np.ndarray:
        if user_id is None:
            self.current_user = np.random.choice(list(self.user_traces.keys()))
        else:
            self.current_user = user_id

        self.trace = self.user_traces[self.current_user]
        self.step_index = 0
        self.remaining_budget = self.time_budget

        # State columns: [attempts, successes, time_since_last_review (hrs), historical_accuracy]
        self.state = np.zeros((self.num_items, 4))
        self.state[:, 2] = 24.0  # default time_since_last_review
        self.state[:, 3] = self.item_features["correctness_rate"].values

        self.last_timestamp = {item_id: None for item_id in self.item_ids}
        self.current_time = None

        return self.state

    def get_available_actions(self) -> np.ndarray:
        return np.arange(self.num_items)

    def _recall_probability(self, item_idx: int, time_since_hours: float) -> float:
        item_id = self.item_ids[item_idx]
        difficulty = float(self.item_features.loc[item_id].get("difficulty", 0.5))

        base_prob = max(0.1, min(0.9, 1.0 - difficulty))
        decay = np.exp(-self.forgetting_rate * max(time_since_hours, 0.0) / 24.0)
        successes = self.state[item_idx, 1]
        success_boost = min(0.3, successes * 0.05)

        prob = base_prob * decay + success_boost
        return float(np.clip(prob, 0.05, 0.97))

    def _update_time_since(self, current_timestamp: pd.Timestamp):
        for item_id, last_ts in self.last_timestamp.items():
            if last_ts is None:
                continue
            idx = self.item_to_idx[item_id]
            gap_hours = (current_timestamp - last_ts).total_seconds() / 3600.0
            self.state[idx, 2] = gap_hours

    def step(self, action: int):
        if action < 0 or action >= self.num_items:
            raise ValueError(f"Invalid action: {action}. Must be in [0, {self.num_items - 1}]")

        if self.step_index >= len(self.trace) or self.remaining_budget <= 0:
            return self.state, 0, True, {"done": True}

        event = self.trace[self.step_index]
        timestamp = event.get("timestamp")
        delta = event.get("delta")

        if self.current_time is None and timestamp is not None:
            self.current_time = timestamp

        true_item_id = self.item_ids[action]
        item_idx = action

        last_ts = self.last_timestamp.get(true_item_id)
        if last_ts is None:
            gap_hours = float(delta) / 3600.0 if delta is not None else 24.0
        else:
            gap_hours = (timestamp - last_ts).total_seconds() / 3600.0 if timestamp is not None else 24.0

        prob = self._recall_probability(item_idx, gap_hours)
        recalled = np.random.random() < prob

        self.state[item_idx, 0] += 1
        if recalled:
            self.state[item_idx, 1] += 1

        attempts = self.state[item_idx, 0]
        successes = self.state[item_idx, 1]
        self.state[item_idx, 3] = successes / attempts if attempts > 0 else 0.0

        if timestamp is not None:
            self.last_timestamp[true_item_id] = timestamp
            self._update_time_since(timestamp)
            self.current_time = timestamp
        else:
            self.last_timestamp[true_item_id] = None

        reward = +1 if recalled else -1

        self.remaining_budget -= 1
        self.step_index += 1

        done = (self.remaining_budget <= 0) or (self.step_index >= len(self.trace))

        info = {
            "chosen_item_id": true_item_id,
            "recalled": recalled,
            "prob": prob,
            "timestamp": timestamp,
            "step": self.step_index
        }

        return self.state, reward, done, info

    def render(self):
        print(f"User: {self.current_user}, Step: {self.step_index}/{len(self.trace)}")
        print(f"Remaining Budget: {self.remaining_budget}")
