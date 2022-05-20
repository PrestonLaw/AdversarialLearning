# Partially taken from https://github.com/pytorch/examples/blob/main/reinforcement_learning/actor_critic.py
# Continuous CartPole implementation from https://gist.github.com/iandanforth/e3ffb67cf3623153e968f2afdfb01dc8

import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.distributions import Normal

from continuous_cartpole import ContinuousCartPoleEnv

import pdb



gamma = 0.99
log_interval = 100



env = ContinuousCartPoleEnv()



SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])



class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        # CartPole has 4 parameters
        self.fc1 = nn.Linear(4, 128)
        # Continuous CartPole has 1 action, continuous float32 from [-1,1]
        self.actor = nn.Linear(128, 1)
        # Critics always have 1 output
        self.critic = nn.Linear(128, 1)
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_mean = F.softmax(self.actor(x), dim=-1)
        state_values = self.critic(x)
        return action_mean, state_values



model = ActorCritic()
optimizer = optim.Adam(model.parameters(), lr=3e-2)
eps = np.finfo(np.float32).eps.item()



def select_action(state):
    state = torch.from_numpy(state).float()
    action_mean, state_value = model(state)

    # Difference here, we can't use a Categorical distribution because the action space is continuous.
    #pdb.set_trace()
    m = Normal(action_mean, (min(1-action_mean, action_mean+1)/3))
    # TODO: Find something that won't select out of the Box bounds, or just clip it.
    np.clip(m, np.float32(-1.0), np.float32(1.0))

    # Sample action from distribution
    action = m.sample()

    # Save to action buffer
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

    # Return the action to take.
    # TODO: Change?
    return action.item()



# Scraped directly.
def finish_episode():
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    R = 0
    saved_actions = model.saved_actions
    policy_losses = [] # list to save actor (policy) loss
    value_losses = [] # list to save critic (value) loss
    returns = [] # list to save the true values

    # calculate the true value using rewards returned from the environment
    for r in model.rewards[::-1]:
        # calculate the discounted value
        R = r + gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()

        # calculate actor (policy) loss 
        policy_losses.append(-log_prob * advantage)

        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

    # reset gradients
    optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    # perform backprop
    loss.backward()
    optimizer.step()

    # reset rewards and action buffer
    del model.rewards[:]
    del model.saved_actions[:]



def main():
    running_reward = 10

    # run inifinitely many episodes
    for i_episode in count(1):

        # reset environment and episode reward
        state = env.reset()
        ep_reward = 0

        # for each episode, only run 9999 steps so that we don't 
        # infinite loop while learning
        for t in range(1, 10000):

            # select action from policy
            action = select_action(state)

            # take the action
            #pdb.set_trace()
            # EDIT: Checking whether an action is part of a Box action space
            # requires the action to be a float32 nparray of shape (1,).
            action = np.reshape(np.asarray(action), (1,))
            state, reward, done, _ = env.step(np.float32(action))

            #if args.render:
            #env.render()

            model.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # perform backprop
        finish_episode()

        # log results
        if i_episode % log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))

        # check if we have "solved" the cart pole problem
        #pdb.set_trace()
        #if running_reward > env.spec.reward_threshold:
        if running_reward > 100:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()