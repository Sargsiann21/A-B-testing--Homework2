from abc import ABC, abstractmethod
from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

class BanditStrategy(ABC):
    @abstractmethod
    def __init__(self, probabilities, trials):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull_arm(self, arm_index):
        pass

    @abstractmethod
    def update_values(self, arm_index, reward):
        pass

    @abstractmethod
    def run_experiment(self):
        pass

    @abstractmethod
    def generate_report(self):
        pass


class EpsilonGreedyStrategy(BanditStrategy):
    def __init__(self, probabilities, trials=20000):
        self.probabilities = probabilities
        self.trials = trials
        self.epsilon = 1.0
        self.num_arms = len(probabilities)
        self.arm_pulls = [0] * self.num_arms
        self.estimated_values = [0.0] * self.num_arms
        self.collected_rewards = []
        self.calculated_regrets = []

    def __repr__(self):
        return "Epsilon-Greedy Strategy"

    def pull_arm(self, arm_index):
        return 1 if random.random() < self.probabilities[arm_index] else 0

    def update_values(self, arm_index, reward):
        self.arm_pulls[arm_index] += 1
        self.estimated_values[arm_index] += (reward - self.estimated_values[arm_index]) / self.arm_pulls[arm_index]

    def run_experiment(self):
        max_possible_reward = max(self.probabilities) * self.trials
        for t in range(1, self.trials + 1):
            self.epsilon = 1 / t
            if random.random() < self.epsilon:
                chosen_arm = random.randint(0, self.num_arms - 1)
            else:
                chosen_arm = np.argmax(self.estimated_values)

            reward = self.pull_arm(chosen_arm)
            self.update_values(chosen_arm, reward)
            self.collected_rewards.append(reward)
            self.calculated_regrets.append(max_possible_reward - sum(self.collected_rewards))

    def generate_report(self):
        average_reward = np.mean(self.collected_rewards)
        average_regret = np.mean(self.calculated_regrets)
        logger.info(f"Average Reward: {average_reward:.4f}")
        logger.info(f"Average Regret: {average_regret:.4f}")
        result_data = {
            "Trial": range(len(self.collected_rewards)),
            "Reward": self.collected_rewards,
            "Strategy": ["EpsilonGreedy"] * len(self.collected_rewards),
        }
        pd.DataFrame(result_data).to_csv("epsilon_greedy_results.csv", index=False)


class ThompsonSamplingStrategy(BanditStrategy):
    def __init__(self, probabilities, trials=20000):
        self.probabilities = probabilities
        self.trials = trials
        self.num_arms = len(probabilities)
        self.success_counts = [1] * self.num_arms
        self.failure_counts = [1] * self.num_arms
        self.collected_rewards = []
        self.calculated_regrets = []

    def __repr__(self):
        return "Thompson Sampling Strategy"

    def pull_arm(self, arm_index):
        return 1 if random.random() < self.probabilities[arm_index] else 0

    def update_values(self, arm_index, reward):
        if reward == 1:
            self.success_counts[arm_index] += 1
        else:
            self.failure_counts[arm_index] += 1

    def run_experiment(self):
        max_possible_reward = max(self.probabilities) * self.trials
        for _ in range(self.trials):
            sampled_values = [
                np.random.beta(self.success_counts[i], self.failure_counts[i]) for i in range(self.num_arms)
            ]
            chosen_arm = np.argmax(sampled_values)
            reward = self.pull_arm(chosen_arm)
            self.update_values(chosen_arm, reward)
            self.collected_rewards.append(reward)
            self.calculated_regrets.append(max_possible_reward - sum(self.collected_rewards))

    def generate_report(self):
        average_reward = np.mean(self.collected_rewards)
        average_regret = np.mean(self.calculated_regrets)
        logger.info(f"Average Reward: {average_reward:.4f}")
        logger.info(f"Average Regret: {average_regret:.4f}")
        result_data = {
            "Trial": range(len(self.collected_rewards)),
            "Reward": self.collected_rewards,
            "Strategy": ["ThompsonSampling"] * len(self.collected_rewards),
        }
        pd.DataFrame(result_data).to_csv("thompson_sampling_results.csv", index=False)


class ResultsVisualizer:
    def plot_cumulative_rewards(self, eg_rewards, ts_rewards):
        plt.plot(np.cumsum(eg_rewards), label="Epsilon Greedy")
        plt.plot(np.cumsum(ts_rewards), label="Thompson Sampling")
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Reward")
        plt.legend()
        plt.title("Cumulative Rewards Over Trials")
        plt.show()

    def plot_cumulative_regrets(self, eg_regrets, ts_regrets):
        plt.plot(np.cumsum(eg_regrets), label="Epsilon Greedy")
        plt.plot(np.cumsum(ts_regrets), label="Thompson Sampling")
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Regret")
        plt.legend()
        plt.title("Cumulative Regrets Over Trials")
        plt.show()


def run_comparison():
    bandit_probs = [0.1, 0.2, 0.3, 0.4]
    epsilon_greedy = EpsilonGreedyStrategy(bandit_probs)
    thompson_sampling = ThompsonSamplingStrategy(bandit_probs)

    logger.info("Executing Epsilon Greedy Strategy")
    epsilon_greedy.run_experiment()
    epsilon_greedy.generate_report()

    logger.info("Executing Thompson Sampling Strategy")
    thompson_sampling.run_experiment()
    thompson_sampling.generate_report()

    visualizer = ResultsVisualizer()
    visualizer.plot_cumulative_rewards(epsilon_greedy.collected_rewards, thompson_sampling.collected_rewards)
    visualizer.plot_cumulative_regrets(epsilon_greedy.calculated_regrets, thompson_sampling.calculated_regrets)


if __name__ == "__main__":
    run_comparison()
