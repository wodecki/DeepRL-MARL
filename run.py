#!/usr/bin/env python
# coding: utf-8

# # Navigation - Deep Q-Network implementation

from unityagents import UnityEnvironment
from maddpg_agent import Agent
import sys
import random
import torch
import numpy as np
from collections import deque
from parameters import *
import os


# Instantiate the Environment and Agent

env = UnityEnvironment(file_name='Tennis_Linux_NoVis/Tennis.x86_64')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of actions
action_size = brain.vector_action_space_size
print('action size = ', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
agents_no = len(env_info.agents)

try:
    os.mkdir("./models")
except OSError:
    print("'models' dir already exists...")

def maddpg(model_number, TAU, LR_ACTOR, LR_CRITIC, fc_units, n_episodes=700, max_t=1000):
    """ Deep Deterministic Policy Gradients
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
    """

    fc1_units = fc_units[0]
    fc2_units = fc_units[1]

    agent_1 = Agent(state_size, action_size, TAU, LR_ACTOR, LR_CRITIC, fc1_units, fc2_units, random_seed=123)
    agent_2 = Agent(state_size, action_size, TAU, LR_ACTOR, LR_CRITIC, fc1_units, fc2_units, random_seed=345)
    agents = [agent_1, agent_2]

    scores_window = deque(maxlen=100)
    scores = np.zeros(agents_no)
    scores_episode = []
    noise = 2
    noise_reduction = 0.9999

    score_avg = 0
    scores_avg_max = 0

    solved_counter = 0

    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations

        for agent in agents:
            agent.reset()

        scores = np.zeros(agents_no)

        for t in range(max_t):
            actions = np.array([agents[i].act(states[i],noise) for i in range(agents_no)])
            noise *= noise_reduction

            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            for i in range(agents_no):
                agents[i].step(t,states[i], actions[i], rewards[i], next_states[i], dones[i])

            states = next_states
            scores += rewards

            print('\rEpisode {}\tMax. score: {:.3f}'
                .format(i_episode, np.max(scores)), end="")
            if np.any(dones):
                break

        score = np.max(scores)
        scores_window.append(score)
        scores_episode.append(score)

        score_avg = np.mean(scores_window)

        if score_avg > scores_avg_max:
            scores_avg_max = score_avg
            if scores_avg_max >=0.5:
                    # save the episode model solved the environment
                    solved_counter += 1
                    if solved_counter == 1:
                        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(len(scores_episode), score_avg))
                        with open('./models/models_solved.txt', 'a') as solved:
                            solved.writelines('{}, {}, {:.3f} \n'.format(model_number, i_episode, score_avg))
                            solved.flush()

                    # store the best model which solved the environment
                    # print('I am saving a model with scores_avg_max =  ', scores_avg_max)
                    torch.save(agents[0].actor_local.state_dict(), './models/checkpoint_actor1_'+str(model_number)+'.pth')
                    torch.save(agents[1].actor_local.state_dict(), './models/checkpoint_actor2_'+str(model_number)+'.pth')
                    #break #comment this line if you want to continue computations even if the environment is solved

        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.3f}\tMax avg. score till now: {:.3f}'\
                  .format(i_episode, score, score_avg, scores_avg_max))

        with open('results.txt', 'a') as output:
            output.writelines(\
            '{}, {},    ,{:.3f}, {:.3f}, {:.3f},    , {}, {}, {}, {} \n'.format(
            model_number, i_episode, score, score_avg, scores_avg_max, \
            TAU, LR_ACTOR, LR_CRITIC, fc_units))
            output.flush()

    return scores_episode

model_number = 1

total_no_models = len(r_fc_units)*len(r_LR_ACTOR)*len(r_LR_CRITIC)*len(r_TAU)
print('Total number of models to test: ', total_no_models)


for TAU in r_TAU:
    for fc_units in r_fc_units:
        for LR_ACTOR in r_LR_ACTOR:
            for LR_CRITIC in r_LR_CRITIC:
                maddpg(model_number, TAU, LR_ACTOR, LR_CRITIC, fc_units, n_episodes=700, max_t=1000)
                model_number +=1
