"""Implements k-armed testbed with greedy and epsilon-greedy agents."""

import numpy as np
import pandas as pd


class Testbed:
    """Testbed class to set up environment for agents.

       Can specify reward values or input mean and var for random
       generation.

       truerewards = np.array if given
    """

    def __init__(self, arms=int, truerewards=None, mean=0, var=1, playvar=1):
        self.k = arms
        randomised = np.random.normal(mean, var, arms)
        self.truerewards = (truerewards if truerewards is not None
                            else randomised)
        self.playvar = playvar

    def playreward(self, i):
        """Actual reward given when option i selected."""
        return np.random.normal(self.truerewards[i], self.playvar)


class Agent:
    """Basic Agent class.

       Specify how long
       the run should be.
    """

    def __init__(self, runlength):
        self.runlength = runlength
        #self.name = name


class Greedy(Agent):
    """Agent employing greedy strategy."""

    def __init__(self, runlength, testbed):
        super().__init__(runlength, testbed)

    def play(self, testbed=Testbed, initialvals=None):
        """Do a test run.

            When multiple values are the max
            it just takes the first one.
        """
        results = []
        timesselected = np.zeros(testbed.k)
        optimalreward = testbed.truerewards.argmax()
        optimalrewardselected = []
        sample_average = initialvals if initialvals is not None else np.zeros(testbed.k)
        for x in range(self.runlength):
            i = np.argmax(sample_average)
            if i == optimalreward:
                optimalrewardselected.append(1)
            else:
                optimalrewardselected.append(0)
            reward = testbed.playreward(i)
            results.append(reward)
            timesselected[i] += 1
            sample_average[i] = rewardsestimate(sample_average[i],
                                                reward, timesselected[i])
        return results, optimalrewardselected


class EpsilonGreedy(Agent):
    """Agent employing epsilon-greedy strategy."""

    def __init__(self, runlength, epsilon):
        super().__init__(runlength)
        self.epsilon = epsilon

    def play(self, testbed=Testbed, initialvals=None):
        """Chooses a random option with probability epsilon, otherwise
           uses greedy strategy.
        """
        results = []
        timesselected = np.zeros(testbed.k)
        optimalreward = testbed.truerewards.argmax()
        optimalrewardselected = []
        sample_average = initialvals if initialvals is not None else np.zeros(testbed.k)
        for x in range(self.runlength):
            if self.epsilon > np.random.uniform(0, 1):
                i = np.random.randint(0, high=testbed.k)
            else:
                i = np.argmax(sample_average)
            if i == optimalreward:
                optimalrewardselected.append(1)
            else:
                optimalrewardselected.append(0)
            reward = testbed.playreward(i)
            results.append(reward)
            timesselected[i] += 1
            sample_average[i] = rewardsestimate(sample_average[i],
                                                reward, timesselected[i])
        return results, optimalrewardselected

    def multsim(self, testbed, runnum=100):
        rewardsmat = []
        optimalsmat = []
        for run in range(runnum):
            res = self.play(testbed)
            rewardsmat.append(res[0])
            optimalsmat.append(res[1])
        run_rewards_df = pd.DataFrame(rewardsmat).T
        self.average_rewards = run_rewards_df.mean(axis=1)
        optimal_action_df = pd.DataFrame(optimalsmat).T
        self.optimal_action_percent = optimal_action_df.mean(axis=1) * 100


def rewardsestimate(qn, rn, n):
    qn += (rn-qn)/n
    return qn
