from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf
# import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
import time

import os
import gym
import matplotlib.pyplot as pl
import numpy as np
from tensorflow.keras import optimizers as opt
from tensorflow.keras import models as nn
from tensorflow.keras import layers as ly

import grid2op
import numpy as np
import pandas as pd
import time
import pickle
import random

np.random.seed(19)

env = grid2op.make("/home/g.krishnamoorthy/data_grid2op/l2rpn_neurips_2020_track2_small")
data = pd.read_csv('track2_output.csv')
data = data.to_numpy()
data = data[:, 0:2]
data = data.tolist()
random.shuffle(data)

batch_size = 10
chronic_batch_list = []
for data_index in range(np.int(len(data) / batch_size)):
    chronic_batch_list.append([])
    chronic_batch_list[data_index] = data[data_index * batch_size:(data_index + 1) * batch_size]

line_or_to_subid = env.action_space.line_or_to_subid
line_ex_to_subid = env.action_space.line_ex_to_subid

sub_line_count = []
for i in range(118):
    count = 0
    for k in range(186):
        if line_or_to_subid[k] == i:
            count += 1
        if line_ex_to_subid[k] == i:
            count += 1
    sub_line_count.append(count)

subid_with_4lines = np.where(np.array(sub_line_count) == 4)
subid_with_5lines = np.where(np.array(sub_line_count) == 5)
subid_with_6lines = np.where(np.array(sub_line_count) == 6)
subid_with_7lines = np.where(np.array(sub_line_count) == 7)
subid_for_topo_action = np.array(
    np.sort(np.concatenate((subid_with_4lines, subid_with_5lines, subid_with_6lines, subid_with_7lines), axis=1))[0])

# print(subid_for_topo_action)

topology_choice = []
subid_for_topo_action_map = []
from itertools import combinations

for i in range(len(subid_for_topo_action)):

    line_count = sub_line_count[subid_for_topo_action[i]]
    # print("-----------------------------------------------------------------------------------------------")
    # print("Substation ID:",subid_for_topo_action[i])
    # print("-----------------------------------------------------------------------------------------------")
    # print("Number of lines connected to sub-id:",subid_for_topo_action[i],"is",line_count)

    # print("Possible Topology for sub-id:",subid_for_topo_action[i],"(considering only lines (load/Gen are not considered)) are:")
    var = np.arange(line_count)
    if line_count < 6:
        comb = combinations(var, 2)
        internal_count = 0
        for k in list(comb):
            internal_count += 1
            target_topology = np.ones(env.sub_info[subid_for_topo_action[i]], dtype=int)
            target_topology[k[0]] = 2
            target_topology[k[1]] = 2
            # print("Sequence",internal_count,"is:",target_topology)
            reconfig_sub = env.action_space(
                {"set_bus": {"substations_id": [(subid_for_topo_action[i], target_topology)]}})
            # print(reconfig_sub)
            topology_choice.append(target_topology)
            subid_for_topo_action_map.append(subid_for_topo_action[i])
        target_topology = np.ones(env.sub_info[subid_for_topo_action[i]], dtype=int)
        topology_choice.append(target_topology)
        subid_for_topo_action_map.append(subid_for_topo_action[i])

    if line_count >= 6:
        comb = combinations(var, 2)
        internal_count = 0
        for k in list(comb):
            target_topology = np.ones(env.sub_info[subid_for_topo_action[i]], dtype=int)
            target_topology[k[0]] = 2
            target_topology[k[1]] = 2
            # print(target_topology)
            reconfig_sub = env.action_space(
                {"set_bus": {"substations_id": [(subid_for_topo_action[i], target_topology)]}})
            # print(reconfig_sub)
            topology_choice.append(target_topology)
            subid_for_topo_action_map.append(subid_for_topo_action[i])
        comb = combinations(var, 3)
        for k in list(comb):
            target_topology = np.ones(env.sub_info[subid_for_topo_action[i]], dtype=int)
            target_topology[k[0]] = 2
            target_topology[k[1]] = 2
            target_topology[k[2]] = 2
            # print(target_topology)
            reconfig_sub = env.action_space(
                {"set_bus": {"substations_id": [(subid_for_topo_action[i], target_topology)]}})
            # print(reconfig_sub)
            topology_choice.append(target_topology)
            subid_for_topo_action_map.append(subid_for_topo_action[i])
        target_topology = np.ones(env.sub_info[subid_for_topo_action[i]], dtype=int)
        topology_choice.append(target_topology)
        subid_for_topo_action_map.append(subid_for_topo_action[i])


# print(len(topology_choice))
def merge_3(list1, list2, list3):
    merged_list = [(list1[i], list2[i], list3[i]) for i in range(0, len(list1))]
    return merged_list


def merge(list1, list2):
    merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))]
    return merged_list


list1 = subid_for_topo_action_map
list2 = topology_choice
topology_action_map = merge(list1, list2)

line_sub_list = []
for line_id, (ori, ext) in enumerate(zip(line_or_to_subid, line_ex_to_subid)):
    line_sub_list.append([line_id, ori, ext])

sub_line_ID_map = []
for index in range(118):
    sub_line_ID_map.append([])
    for k in range(186):
        if line_or_to_subid[k] == index:
            sub_line_ID_map[index].append(k)
        if line_ex_to_subid[k] == index:
            sub_line_ID_map[index].append(k)

area_1 = [0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 18, 19, 116]
area_2 = [7, 8, 9, 15, 16, 17, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31, 112, 113, 114]
area_3 = [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
area_4 = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 115]
area_5 = [23, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 81, 82, 83, 84, 85, 86, 87, 88, 89, 117]
area_6 = [79, 80, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]

line_id_area_map = []
for line_idx in range(len(line_sub_list)):
    line_id_area_map.append([])
    if line_sub_list[line_idx][1] in area_1:
        line_id_area_map[line_idx].append(1)
    elif line_sub_list[line_idx][2] in area_1:
        line_id_area_map[line_idx].append(1)
    if line_sub_list[line_idx][1] in area_2:
        line_id_area_map[line_idx].append(2)
    elif line_sub_list[line_idx][2] in area_2:
        line_id_area_map[line_idx].append(2)
    if line_sub_list[line_idx][1] in area_3:
        line_id_area_map[line_idx].append(3)
    elif line_sub_list[line_idx][2] in area_3:
        line_id_area_map[line_idx].append(3)
    if line_sub_list[line_idx][1] in area_4:
        line_id_area_map[line_idx].append(4)
    elif line_sub_list[line_idx][2] in area_4:
        line_id_area_map[line_idx].append(4)
    if line_sub_list[line_idx][1] in area_5:
        line_id_area_map[line_idx].append(5)
    elif line_sub_list[line_idx][2] in area_5:
        line_id_area_map[line_idx].append(5)
    if line_sub_list[line_idx][1] in area_6:
        line_id_area_map[line_idx].append(6)
    elif line_sub_list[line_idx][2] in area_6:
        line_id_area_map[line_idx].append(6)

area_1_line = []
for k in area_1:
    # print(sub_line_ID_map[k])
    area_1_line.extend(sub_line_ID_map[k])
area_1_line = list(set(area_1_line))
# print(area_1_line)

area_2_line = []
for k in area_2:
    # print(sub_line_ID_map[k])
    area_2_line.extend(sub_line_ID_map[k])
area_2_line = list(set(area_2_line))
# print(area_2_line)

area_3_line = []
for k in area_3:
    # print(sub_line_ID_map[k])
    area_3_line.extend(sub_line_ID_map[k])
area_3_line = list(set(area_3_line))
# print(area_3_line)

area_4_line = []
for k in area_4:
    # print(sub_line_ID_map[k])
    area_4_line.extend(sub_line_ID_map[k])
area_4_line = list(set(area_4_line))
# print(area_4_line)

area_5_line = []
for k in area_5:
    # print(sub_line_ID_map[k])
    area_5_line.extend(sub_line_ID_map[k])
area_5_line = list(set(area_5_line))
# print(area_5_line)

area_6_line = []
for k in area_6:
    # print(sub_line_ID_map[k])
    area_6_line.extend(sub_line_ID_map[k])
area_6_line = list(set(area_6_line))
# print(area_6_line)


switch_line_action_area_map = [area_1_line, area_2_line, area_3_line, area_4_line, area_5_line, area_6_line]

zone_1 = [4, 10, 11, 14, 18]
zone_2 = [16, 26, 31, 29, 22]
zone_3 = [33, 36, 39, 41]
zone_4 = [53, 55, 60, 61, 63, 64, 65, 67, 68]
zone_5 = [69, 74, 84, 88]
zone_6 = [93, 95, 104, 102, 109]

topology_sub_id_zone_1 = list()
topology_sub_id_zone_2 = list()
topology_sub_id_zone_3 = list()
topology_sub_id_zone_4 = list()
topology_sub_id_zone_5 = list()
topology_sub_id_zone_6 = list()
topology_action_zone_1 = list()
topology_action_zone_2 = list()
topology_action_zone_3 = list()
topology_action_zone_4 = list()
topology_action_zone_5 = list()
topology_action_zone_6 = list()

area_1_actionID_list = list()
area_2_actionID_list = list()
area_3_actionID_list = list()
area_4_actionID_list = list()
area_5_actionID_list = list()
area_6_actionID_list = list()

for i in zone_1:
    for k in range(len(topology_action_map)):
        if topology_action_map[k][0] == i:
            topology_action_zone_1.append(topology_action_map[k][1])
            topology_sub_id_zone_1.append(topology_action_map[k][0])
            area_1_actionID_list.append(k)
topology_action_map_zone_1 = merge_3(topology_sub_id_zone_1, topology_action_zone_1, area_1_actionID_list)
# topology_action_map_zone_1 = merge(topology_action_map_zone_1,area_1_actionID_list)
for i in zone_2:
    for k in range(len(topology_action_map)):
        if topology_action_map[k][0] == i:
            topology_action_zone_2.append(topology_action_map[k][1])
            topology_sub_id_zone_2.append(topology_action_map[k][0])
            area_2_actionID_list.append(k)
topology_action_map_zone_2 = merge_3(topology_sub_id_zone_2, topology_action_zone_2, area_2_actionID_list)
# topology_action_map_zone_2 = merge(topology_action_map_zone_2,area_2_actionID_list)
for i in zone_3:
    for k in range(len(topology_action_map)):
        if topology_action_map[k][0] == i:
            topology_action_zone_3.append(topology_action_map[k][1])
            topology_sub_id_zone_3.append(topology_action_map[k][0])
            area_3_actionID_list.append(k)
topology_action_map_zone_3 = merge_3(topology_sub_id_zone_3, topology_action_zone_3, area_3_actionID_list)
# topology_action_map_zone_3 = merge(topology_action_map_zone_3,area_3_actionID_list)
for i in zone_4:
    for k in range(len(topology_action_map)):
        if topology_action_map[k][0] == i:
            topology_action_zone_4.append(topology_action_map[k][1])
            topology_sub_id_zone_4.append(topology_action_map[k][0])
            area_4_actionID_list.append(k)
topology_action_map_zone_4 = merge_3(topology_sub_id_zone_4, topology_action_zone_4, area_4_actionID_list)
# topology_action_map_zone_4 = merge(topology_action_map_zone_4,area_4_actionID_list)
for i in zone_5:
    for k in range(len(topology_action_map)):
        if topology_action_map[k][0] == i:
            topology_action_zone_5.append(topology_action_map[k][1])
            topology_sub_id_zone_5.append(topology_action_map[k][0])
            area_5_actionID_list.append(k)
topology_action_map_zone_5 = merge_3(topology_sub_id_zone_5, topology_action_zone_5, area_5_actionID_list)
# topology_action_map_zone_5 = merge(topology_action_map_zone_5,area_5_actionID_list)
for i in zone_6:
    for k in range(len(topology_action_map)):
        if topology_action_map[k][0] == i:
            topology_action_zone_6.append(topology_action_map[k][1])
            topology_sub_id_zone_6.append(topology_action_map[k][0])
            area_6_actionID_list.append(k)
topology_action_map_zone_6 = merge_3(topology_sub_id_zone_6, topology_action_zone_6, area_6_actionID_list)
# topology_action_map_zone_6 = merge(topology_action_map_zone_6,area_6_actionID_list)

topo_action_area_map = [topology_action_map_zone_1, topology_action_map_zone_2, topology_action_map_zone_3,
                        topology_action_map_zone_4, topology_action_map_zone_5, topology_action_map_zone_6]


# action_ID_master_list = [area_1_actionID_list,area_2_actionID_list,area_3_actionID_list,area_4_actionID_list,area_5_actionID_list,area_6_actionID_list]


def max_loaded_line_info(obs, line_id_area_map):
    max_loaded_lineID = np.argmax(obs.rho)
    areaID = line_id_area_map[max_loaded_lineID]
    return areaID


def combined_action(topo_action_area_map, switch_line_action_area_map, areaID):
    topo_config_possible = list()
    switch_line_possible = list()

    for id in areaID:
        topo_config_possible.extend(topo_action_area_map[areaID[0] - 1])
        switch_line_possible.extend(switch_line_action_area_map[areaID[0] - 1])

    return topo_config_possible, switch_line_possible


def action_topo_config(topology_possible, action_id):
    if action_id == len(topology_possible):
        action = env.action_space({})
        UniqueActionID = 652
        # print('It is better to take no action')
    else:
        UniqueActionID = topology_possible[np.int(action_id)][2]
        target_topology = topology_possible[np.int(action_id)][1]
        subid_for_topo_action = topology_possible[np.int(action_id)][0]

        action = env.action_space({"set_bus": {"substations_id": [(subid_for_topo_action, target_topology)]}})
    # print(id_line)
    # print("UniqueActionID:",UniqueActionID)
    return action, UniqueActionID


def action_switch_line(switch_line_possible, ID):
    id_line = np.int(switch_line_possible[np.int(ID)])
    action = env.action_space({"change_line_status": [(id_line)]})
    return action, id_line


def best_action(topo_config_possible, switch_line_possible):
    sim_next_rho_topo_config = list()
    exception_topo_config = list()
    sim_next_rho_switch_line = list()
    exception_switch_line = list()
    for i in range(len(topo_config_possible)):
        action_id = i
        action_sim, _ = action_topo_config(topo_config_possible, action_id)
        obs_sim = obs.simulate(action_sim, time_step=1)
        max_sim_rho = np.max(obs_sim[0].rho)
        sim_next_rho_topo_config.append(max_sim_rho)
        exception_topo_config.append(obs_sim[3]['exception'])
    ### add no action simulation

    action_sim = env.action_space({})
    obs_sim = obs.simulate(action_sim, time_step=1)
    sim_next_rho_topo_config.append(np.max(obs_sim[0].rho))
    exception_topo_config.append(obs_sim[3]['exception'])

    sim_next_rho_topo_config = np.array(sim_next_rho_topo_config)
    where_are_NaNs = np.isnan(sim_next_rho_topo_config)
    sim_next_rho_topo_config[where_are_NaNs] = 10

    topo_rho_sim_min = np.min(sim_next_rho_topo_config)
    # switch_line_sim_min = np.min(sim_next_rho_switch_line)

    for i in range(len(switch_line_possible)):
        ID = i
        action_sim, _ = action_switch_line(switch_line_possible, ID)
        obs_sim = obs.simulate(action_sim, time_step=1)
        max_sim_rho = np.max(obs_sim[0].rho)
        sim_next_rho_switch_line.append(max_sim_rho)
        exception_switch_line.append(obs_sim[3]['exception'])
    sim_next_rho_switch_line = np.array(sim_next_rho_switch_line)
    where_are_NaNs = np.isnan(sim_next_rho_switch_line)
    sim_next_rho_switch_line[where_are_NaNs] = 10

    switch_line_sim_min = np.min(sim_next_rho_switch_line)
    # print(topo_rho_sim_min)
    # print(switch_line_sim_min)

    if (topo_rho_sim_min < switch_line_sim_min):
        pick = np.argmin(sim_next_rho_topo_config)
        # print(exception_topo_config[pick])
        if exception_topo_config[pick] == []:
            action, UniqueActionID = action_topo_config(topo_config_possible, pick)
            # UniqueActionID = pick
            # print('a')
        else:
            action = env.action_space({})
            UniqueActionID = 652
            # print('b')
    elif (topo_rho_sim_min >= switch_line_sim_min):
        pick = np.argmin(sim_next_rho_switch_line)
        # print(pick)
        # print(exception_switch_line[pick])
        # print(switch_line_sim_min)
        if exception_switch_line[pick] == []:
            action, id_line = action_switch_line(switch_line_possible, pick)
            UniqueActionID = id_line + 653
            # print('c')
        else:
            pick = np.argmin(sim_next_rho_topo_config)
            if exception_topo_config[pick] == []:
                action, UniqueActionID = action_topo_config(topo_config_possible, pick)
                # UniqueActionID = pick
                # print('a')
            else:
                action = env.action_space({})
                UniqueActionID = 652
                # print('d')
    # print(exception_topo_config[pick])
    # print('simulated_rho_max:',topo_rho_sim_min)

    # print("UniqueActionID:",pick)
    return action, UniqueActionID


gen_info = pd.read_csv('gen_type.csv')

wind = []
thermal = []
solar = []
hydro = []
nuclear = []

for n_gen in range(len(gen_info)):
    if gen_info['Type'][n_gen] == 'wind':
        wind.append(n_gen)
    if gen_info['Type'][n_gen] == 'thermal':
        thermal.append(n_gen)

    if gen_info['Type'][n_gen] == 'solar':
        solar.append(n_gen)
    if gen_info['Type'][n_gen] == 'hydro':
        hydro.append(n_gen)

    if gen_info['Type'][n_gen] == 'nuclear':
        nuclear.append(n_gen)

# episode_no = 1001
# Total batch size for training
mini_batch_size = 128
# Replay memory's max capacity
replay_mem_size = 100000
# Max time steps allowed for each episode
time_steps = 100
# Discount factor
gamma = 0.99
# Exploration rate's minimum value
min_ep1 = 0.01
# Exploration rate's maximum value
max_ep1 = 0.75

max_ep2 = 0.25
min_ep2 = 0.01
# Exploration decay rate
# decay = 0.9
# Reduction step of exploration rate
reduction_step = 0.005
# Setting the frequency at which the model is being updated
update_freq_model = 15
# Setting the initial exploration rate
ep1 = max_ep1
ep2 = max_ep2
# Saving the rewards for every epsiode
calc_scores = []
# Saving the loss for the training in each time-step
loss = []
# The replay memory stores the values for the states, actions, rewards and next states
experience_replay = []
# Saving the mean of scores for previous 100 episodes
score_mean = []
# Saving the standard deviation for the scores of previous 100 episodes
stdev = []
epi_len = []
# Constructing an environment instance
# env = gym.make("Acrobot-v1")
# env._max_episode_steps = time_steps
dimension_action = 653 + 186
dimension_input = 196

dnn_A = nn.Sequential()
dnn_A.add(ly.Dense(256, input_dim=dimension_input, activation='relu'))
dnn_A.add(ly.Dense(256, activation='relu'))
# =============================================================================
# dnn_A.add(ly.Dense(64, activation = 'relu'))
# dnn_A.add(ly.Dense(128, activation = 'relu'))
# dnn_A.add(ly.Dense(64, activation = 'relu'))
# dnn_A.add(ly.Dense(128, activation = 'relu'))
# dnn_A.add(ly.Dense(64, activation = 'relu'))
# dnn_A.add(ly.Dense(128, activation = 'relu'))
# dnn_A.add(ly.Dense(128, activation = 'relu'))
# =============================================================================
dnn_A.add(ly.Dense(dimension_action, activation='linear'))
dnn_A.compile(optimizer=opt.Adam(), loss='mse')
# dnn_A.summary
dnn_B = dnn_A


def model_update(replay_memory, minibatch_size):
    # global loss
    minibatch = np.random.choice(replay_memory, minibatch_size, replace=True)
    # Extract the state, action, reward, next state, and done from the batch
    state_l = np.array(list(map(lambda x: x['state'], minibatch)))
    action_l = np.array(list(map(lambda x: x['action'], minibatch)))
    reward_l = np.array(list(map(lambda x: x['reward'], minibatch)))

    next_state_l = np.array(list(map(lambda x: x['next_state'], minibatch)))
    done_l = np.array(list(map(lambda x: x['done'], minibatch)))
    # Estimate Q-values of 'next states' using current estimate of weights
    qvals_next_state_l = dnn_A.predict(next_state_l)
    qvals_next_state_l_tar = dnn_B.predict(next_state_l)
    # Estimate Q-values of the 'states' using current estimate of weights
    target_f = dnn_A.predict(state_l)  # includes the other actions, states
    for i, (state, action, reward, qvals_next_state, qvals_next_state_tar, done) in \
            enumerate(zip(state_l, action_l, reward_l, qvals_next_state_l, qvals_next_state_l_tar, done_l)):
        # Get the estimate of current states from the next states
        # print(qvals_next_state_tar)
        if not done:
            target = reward + gamma * qvals_next_state_tar[np.argmax(qvals_next_state)]
        else:
            target = reward
        target_f[i][action] = target
    # Update DNN weights by gradient descent
    hist = dnn_A.fit(state_l, target_f, epochs=1, verbose=0)
    # Store the current loss
    loss.append(hist.history)
    return dnn_A


def gen_mix(prod_p, prod_q):
    feature_1 = np.array(
        [prod_p[wind].sum(), prod_p[thermal].sum(), prod_p[solar].sum(), prod_p[hydro].sum(), prod_p[nuclear].sum()])
    feature_2 = feature_1 / np.sum(np.absolute(feature_1))
    feature_3 = np.array(
        [prod_q[wind].sum(), prod_q[thermal].sum(), prod_q[solar].sum(), prod_q[hydro].sum(), prod_q[nuclear].sum()])
    feature_4 = feature_3 / np.sum(np.absolute(feature_3))
    # print(feature_2)
    # print(feature_4)
    return feature_2, feature_4


def normalize_input(x):
    x_min = np.min(x)
    x_max = np.max(x)
    normalized_x = (x - x_min) / (x_min - x_max)
    return normalized_x


def preprocess_observation(obs):
    prod_p = obs.prod_p
    prod_q = obs.prod_q

    obs_1, obs_2 = gen_mix(prod_p, prod_q)
    l_c = obs.connectivity_matrix()[0:186, 0:186]

    # obs_3 = normalize_input(np.dot(l_c,obs.rho))
    obs_3 = obs.rho
    observation = np.concatenate((obs_1, obs_2, obs_3), axis=None)

    return observation


def exploration_strategy(ep1, ep2):
    action_choice = np.int(np.random.choice(3, 1, p=[ep1, ep2, (1 - ep1 - ep2)]))

    # =============================================================================
    #     if choice == 0:
    #         action_type = 'Rule based'
    #     elif choice == 1:
    #         action_type = 'Random'
    #     elif choice == 2:
    #         action_type = 'NN based'
    # =============================================================================

    return action_choice


def action_table(topology, action_id):
    target_n = topology[np.int(action_id)][1]
    subid_n = topology[np.int(action_id)][0]

    action = env.action_space({"set_bus": {"substations_id": [(subid_n, target_n)]}})
    UniqueActionID = action_id
    obs_sim = obs.simulate(action, time_step=1)

    if obs_sim[3]['is_ambiguous'] is True or obs_sim[3]['is_illegal'] is True:
        action = env.action_space({})
        UniqueActionID = np.int(652)

    return action, UniqueActionID


def reward_function(obs, past_obs, done, move, line_flow_thershold):
    # line_overflow_max5_prev = np.sort(past_obs.rho, axis=None)
    # line_overflow_max5_cur = np.sort(obs.rho, axis=None)

    max_line_flow_past = np.max(past_obs.rho)
    max_line_flow = np.max(obs.rho)

    change_max_line_flow = max_line_flow - max_line_flow_past

    if done is True and move < (Max_inner_step - 1):
        reward = -2000
    else:
        if move < (Max_inner_step - 1):
            if max_line_flow < line_flow_thershold:
                if change_max_line_flow < 0:
                    reward = 0
                elif change_max_line_flow >= 0:
                    reward = -200
            elif max_line_flow >= line_flow_thershold:
                if change_max_line_flow < 0:
                    reward = 0
                elif change_max_line_flow >= 0 and change_max_line_flow < 0.02:
                    reward = -100
                elif change_max_line_flow >= 0.02:
                    reward = -1000
        elif move == (Max_inner_step - 1):
            if max_line_flow < line_flow_thershold:
                reward = 0
            elif max_line_flow > line_flow_thershold and max_line_flow < 1.00:
                reward = -100
            elif max_line_flow > 1.00:
                reward = -2000

    # =============================================================================
    #     if max_line_flow < 0.85:
    #         reward = 500
    #     elif max_line_flow > 0.85 and max_line_flow <= 0.90:
    #         reward = 200
    #     elif max_line_flow > 0.90 and max_line_flow <= 0.95:
    #         reward = 100
    #     elif max_line_flow > 0.95 and max_line_flow < 1.00:
    #         reward = -100
    #     elif max_line_flow > 1.00 and max_line_flow < 1.05:
    #         reward = -200
    #     elif max_line_flow > 1.05:
    #         reward = -500
    # =============================================================================
    # else:
    # reward = -5000

    return reward


from grid2op.Agent import DeltaRedispatchRandomAgent


class DummyAgent(DeltaRedispatchRandomAgent):
    def __init__(self, action_space):
        super().__init__(action_space)


multimix_env = env
my_agent = DummyAgent(multimix_env.action_space)

MAX_STEPS = 500
episode_reward = 0
store_obs = []
store_action = []
multimix_env = env
rl_episode_ct = 0
step_collection = []
episode_score = []

Max_Episode = 1000

Max_inner_step = 5
n = 0

line_flow_thershold = 0.95

for n in range(Max_Episode):
    n = n + 1
    i = 0
    # for i in range(len(chronic_batch_list[0])):
    for i in range(3):
        np.random.seed(211119999)
        # i = 2
        time_spent = 0
        mix = env[chronic_batch_list[0][i][0]]
        # print(mix)
        # mix = env["l2rpn_neurips_2020_track2_x3"]
        multimix_env.reset()

        # Print some info on current episode
        mix_name = mix
        # mix_name = multimix_env.name
        # chronic_name = multimix_env.chronics_handler.get_name()
        chronic_name = chronic_batch_list[0][i][1]
        # chronic_name = 'Scenario_march_43'

        print("Episode [{}] - Mix [{}] - Chronic [{}]".format(i, mix_name, chronic_name))

        done = False
        obs = multimix_env.current_obs

        # reward = 0.0
        step = 0

        while done is False and step < MAX_STEPS:

            if np.max(obs.rho) > line_flow_thershold:
                # if np.max(obs.timestep_overflow) == 2:

                # print('initial_max_line_flow',np.max(obs.rho))
                state = preprocess_observation(obs)
                score = 0
                inner_step = 0
                flag = False

                while done is False and flag is False:

                    # areaID = max_loaded_line_info(obs,line_id_area_map)
                    # topo_config_possible,switch_line_possible = combined_action(topo_action_area_map,switch_line_action_area_map,areaID)

                    qvals_state = dnn_A.predict(state.reshape(1, dimension_input))
                    # print(np.argmax(qvals_state))

                    action_type = exploration_strategy(ep1, ep2)

                    if action_type == 0:
                        areaID = max_loaded_line_info(obs, line_id_area_map)
                        topo_config_possible, switch_line_possible = combined_action(topo_action_area_map,
                                                                                     switch_line_action_area_map,
                                                                                     areaID)

                        action, UniqueActionID = best_action(topo_config_possible, switch_line_possible)
                        # print('action_type', 'rule-based')
                        # print(UniqueActionID)
                    elif action_type == 1:
                        action_id_choice = np.int(np.random.choice(839, 1))
                        if action_id_choice < 652:
                            action, UniqueActionID = action_table(topology_action_map, action_id_choice)
                        elif action_id_choice > 652:
                            id_l = np.int((action_id_choice) - 653)
                            action = env.action_space({"change_line_status": [(id_l)]})
                            UniqueActionID = action_id_choice
                        elif action_id_choice == 652:
                            action = env.action_space({})
                            UniqueActionID = 652
                        # print('action_type', 'random')
                    elif action_type == 2:
                        action_id_choice = np.int(np.argmax(qvals_state));
                        if action_id_choice < 652:
                            action, UniqueActionID = action_table(topology_action_map, action_id_choice)
                        elif action_id_choice > 652:
                            id_l = np.int((action_id_choice) - 653)
                            action = env.action_space({"change_line_status": [(id_l)]})
                            UniqueActionID = action_id_choice
                        elif action_id_choice == 652:
                            action = env.action_space({})
                            UniqueActionID = 652
                        # print('action_type', 'nn-based')

                    # =============================================================================
                    #                 if np.random.random() < ep: action = env.action_space.sample()
                    #                 else: action = np.argmax(qvals_state);
                    # =============================================================================
                    # print(action)
                    # print(UniqueActionID)
                    past_obs = obs

                    obs, _, done, info = multimix_env.step(action)
                    # print(action)
                    # print('max_lineflow',np.max(obs.rho))
                    reward = reward_function(obs, past_obs, done, inner_step, line_flow_thershold)

                    next_state = preprocess_observation(obs)

                    score += reward  # update episode score
                    # print(reward)
                    if len(experience_replay) > replay_mem_size:
                        experience_replay.pop(0)
                    experience_replay.append({"state": state, "action": UniqueActionID, "reward": reward \
                                                 , "next_state": next_state, "done": flag})
                    # Update state
                    state = next_state
                    # Train the DNN using the samples from replay memory
                    dnn_A = model_update(experience_replay, mini_batch_size)

                    inner_step += 1

                    step += 1
                    # print(step)
                    # print(inner_step)
                    # print(UniqueActionID)
                    if inner_step == Max_inner_step or done == True or step == MAX_STEPS:
                        flag = True

                    total_RL_steps = rl_episode_ct * Max_inner_step + inner_step
                    if total_RL_steps % update_freq_model == 0 and total_RL_steps > 0:
                        dnn_B = dnn_A

                # =============================================================================
                #             schedule_timesteps=int(exploration_fraction * episode_no * 100)
                #             ep = exploration(n * 100,schedule_timesteps,exploration_final_eps,initial_ep)
                # =============================================================================
                if ep1 > min_ep1:
                    ep1 -= reduction_step
                # else:
                # ep1 = ep*decay # Decay
                if ep2 > min_ep2:
                    ep2 -= reduction_step

                rl_episode_ct += 1
                print("RL-episode no.", rl_episode_ct)
                print("RL-episode score", score)
                episode_score.append(score)
                if rl_episode_ct % 10 == 0 and rl_episode_ct > 0:
                    plt.plot(episode_score)
            else:

                action = env.action_space({})

                obs, reward, done, info = multimix_env.step(action)

                step += 1
                # print(step)
                # print(action)

        step_collection.append(step)
        print("Final steps for current chronic:", step)
