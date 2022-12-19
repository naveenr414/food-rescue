import pandas as pd
import numpy as np
import torch
import random
from mlp import P2PEngine
from data import P2PEnvironment
import matplotlib.pyplot as plt
from time import time
import seaborn as sns
from plots import *

def get_cumsum(data):
    mean = (np.nancumsum(data[:,:-1], axis=1)/range(1, config['num_epoch']-1)).mean(axis=0)
    std = (np.nancumsum(data[:,:-1], axis=1)/range(1, config['num_epoch']-1)).std(axis=0)
    
    return mean, std

def plot_data(data,labels,file_name):
    fig, ax = plt.subplots()
    if "match" in file_name:
        ax.set_title("Matching regret by epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Matching regret")
    else:
        ax.set_title("Regret by epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Regret")
    
    results = [get_cumsum(i) for i in data]
    results_mean, results_std = zip(*results)
    results_mean = np.stack(results_mean)
    results_std = np.stack(results_std)

    with sns.axes_style("darkgrid"):
        epochs = list(range(101))
        for i in range(len(data)):
            ax.plot(range(1, config['num_epoch']-1), results_mean[i,:], label=labels[i], c=nbc_palette[i])
            ax.fill_between(range(1, config['num_epoch']-1), results_mean[i,:]-results_std[i,:], results_mean[i,:]+results_std[i,:] ,alpha=0.3, facecolor=nbc_palette[i])
        ax.legend()
    fig.savefig(file_name)

config = eval(open("config.txt").read())

def generate_array(config):
    return np.zeros((config['num_trial'], config['num_epoch']-1))

# Matrices storing information on each epoch 
regret_p2p = generate_array(config)
regret_bandit = generate_array(config)
regret_mask = generate_array(config)
regret_all_data = generate_array(config)

regret_match_p2p = generate_array(config)
regret_match_bandit = generate_array(config)
regret_match_mask = generate_array(config)
regret_match_all_data = generate_array(config)


def run_one_round(engine,regret_matrix, match_matrix):
    action, _time = engine.p2p_an_epoch(data_loader, test_feature, epoch_id=epoch)
    matches = engine.get_matches(action,test_label,env._m)
    ro, rb = env.get_reward(action)
    match_reward = env.get_match_reward(matches)
    engine.update_bandit(ro,rb)
    regret_matrix[trial_idx, epoch-1] = (ro.sum()+rb.sum())-best_reward.sum()
    match_matrix[trial_idx, epoch-1] = best_match_reward - match_reward
    
    return ro, rb, match_reward

for trial_idx in range(config['num_trial']):
    env = P2PEnvironment(config, seed=trial_idx * 10)
    engine_p2p = P2PEngine(env, config, pure_bandit=False, predict_mask=False)
    engine_mask = P2PEngine(env, config, pure_bandit=False, predict_mask=True)
    engine_all_data = P2PEngine(env, config, pure_bandit=False, predict_mask=True,use_all_data=True)
    engine_bandit = P2PEngine(env, config, pure_bandit=True, predict_mask=False)   

    start_time = time()
    
    for epoch in range(1, config['num_epoch']):
        data_loader = env.get_data_loader()
        test_feature, test_label = env.get_new_data(epoch)
        
        best_action = engine_p2p.p2p_known_mu(test_label, env._m)
        best_reward_optimization, best_reward_bandit = env.get_reward(best_action)
        best_reward = best_reward_optimization + best_reward_bandit
        best_matches = engine_p2p.get_matches(best_action,test_label, env._m)
        best_match_reward = env.get_match_reward(best_matches)
        
        run_one_round(engine_bandit, regret_bandit, regret_match_bandit)
        run_one_round(engine_p2p, regret_p2p, regret_match_p2p)
        run_one_round(engine_mask, regret_mask, regret_match_mask)
        run_one_round(engine_all_data, regret_all_data, regret_match_all_data)

        env.add_to_data_loader()

        if epoch == config['num_epoch'] - 1:
            end_time = time()
            print('Epoch {} starts !'.format(epoch))
            print('-' * 80)
            
            file_basename = 'runs/n{}m{}d{}capacity{}range{}task{}ns{}trial{}regret_'.format(config['points_per_iter'], 
            config['feature_dim'], config['label_dim'],config['mean_volunteer_attributes'][0],config['mean_volunteer_attributes'][1],config['mean_task_attributes'][0],config['num_sources'],trial_idx)
            
            for algorithm, dataset in zip(['p2p','bandit','mask','alldata'], 
                                          [regret_p2p,regret_bandit,regret_mask,regret_all_data]):
                file_name = file_basename + algorithm + ".npy"
                np.save(file_name,dataset)

nbc_palette = [[0,0,0], [230/255,159/255,0], [86/255,180/255,233/255], [0,158/255,115/255],
          [213/255,94/255,0], [0,114/255,178/255],[213/255,94/255,0/255],[204/255,121/255,167/255]]
sns.palplot(sns.color_palette(nbc_palette))

file_name = 'figures/n{}m{}d{}capacity{}range{}task{}ns{}_regret.png'.format(config['points_per_iter'], 
            config['feature_dim'], config['label_dim'],config['mean_volunteer_attributes'][0],config['mean_volunteer_attributes'][1],config['mean_task_attributes'][0],config['num_sources'])
data = [regret_p2p,regret_mask,regret_all_data,regret_bandit]
labels = ['PROOF', 'PROOF+Mask','PROOF+Mask+All Data','Vanilla OFU']
plot_data(data,labels,file_name)

file_name = 'figures/n{}m{}d{}capacity{}range{}task{}ns{}_match.png'.format(config['points_per_iter'], 
            config['feature_dim'], config['label_dim'],config['mean_volunteer_attributes'][0],config['mean_volunteer_attributes'][1],config['mean_task_attributes'][0],config['num_sources'])
data = [regret_match_p2p,regret_match_mask,regret_match_all_data,regret_match_bandit]
labels = ['PROOF', 'PROOF+Mask','PROOF+Mask+All Data','Vanilla OFU']
plot_data(data,labels,file_name)
