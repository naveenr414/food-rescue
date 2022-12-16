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

config = {'alias': 'mlp_factor8neg4_bz256_166432168_pretrain_reg_0.0000001',
              'learning_alg': 'OLS',
              'num_trial': 10,
              'points_per_iter': 20,
              'feature_dim': 20,
              'label_dim': 5,
              'label_max': 2, 
              'epsilon': 1e-1,
              'eta': 1e-4,
              'KA_norm': 10,
              'delta': 1e-1,
              'num_epoch': 21,
              'learn_iter': 500,
              'batch_size': 32,
              'optimizer': 'adam',
              'adam_lr': 1e-3,
              'l2_regularization': 1e-4, 
              'num_sources': 5,
              'num_destinations': 5, 
              'mean_volunteer_attributes': [0.7,0.7],
              'std_volunteer_attributes': [0.1,0.1],
              'mean_task_attributes': [0.6],
              'std_task_attributes': [0.1],
              }

# Matrices storing information on each epoch 
regret_p2p = np.zeros((config['num_trial'], config['num_epoch']-1))
regret_bandit = np.zeros((config['num_trial'], config['num_epoch']-1))
regret_p2p_o = np.zeros((config['num_trial'], config['num_epoch']-1))
regret_p2p_b = np.zeros((config['num_trial'], config['num_epoch']-1))
regret_match_p2p = np.zeros((config['num_trial'], config['num_epoch']-1))
regret_match_bandit = np.zeros((config['num_trial'], config['num_epoch']-1))
time_p2p = np.zeros((config['num_trial'], config['num_epoch']-1))
time_bandit = np.zeros((config['num_trial'], config['num_epoch']-1))

for trial_idx in range(config['num_trial']):
    env = P2PEnvironment(config, seed=trial_idx * 10)
    engine_p2p = P2PEngine(env, config, pure_bandit=False)
    engine_bandit = P2PEngine(env, config, pure_bandit=True)
    start_time = time()
    
    for epoch in range(1, config['num_epoch']):
        data_loader = env.get_data_loader()
        test_feature, test_label = env.get_new_data(epoch)
        
        action_p2p, time_p2p[trial_idx, epoch-1] = engine_p2p.p2p_an_epoch(data_loader, test_feature, epoch_id=epoch)
        action_bandit, time_bandit[trial_idx, epoch-1] = engine_bandit.p2p_an_epoch(data_loader, test_feature, epoch_id=epoch)
        
        matches_p2p = engine_p2p.get_matches(action_p2p,test_label, env._m)
        matches_bandit = engine_bandit.get_matches(action_bandit,test_label, env._m)

        best_action = engine_p2p.p2p_known_mu(test_label, env._m)
        best_matches = engine_p2p.get_matches(best_action,test_label, env._m)
        
        ro_p2p, rb_p2p = env.get_reward(action_p2p)
        ro_bandit, rb_bandit = env.get_reward(action_bandit)
        ro_best, rb_best = env.get_reward(best_action)
        
        p2p_match_reward = env.get_match_reward(matches_p2p)
        bandit_match_reward = env.get_match_reward(matches_bandit)
        best_match_reward = env.get_match_reward(best_matches)
            
        engine_p2p.update_bandit(ro_p2p, rb_p2p)
        engine_bandit.update_bandit(ro_bandit, rb_bandit)
        env.add_to_data_loader()
        
        regret_p2p[trial_idx, epoch-1] = ro_p2p.sum() + rb_p2p.sum() - ro_best.sum() - rb_best.sum()
        regret_p2p_o[trial_idx, epoch-1] = ro_p2p.sum() - ro_best.sum()
        regret_p2p_b[trial_idx, epoch-1] = rb_p2p.sum() - rb_best.sum()
        regret_bandit[trial_idx, epoch-1] = ro_bandit.sum() + rb_bandit.sum() - ro_best.sum() - rb_best.sum()
        regret_match_bandit[trial_idx, epoch-1] = best_match_reward - bandit_match_reward
        regret_match_p2p[trial_idx, epoch-1] = best_match_reward - p2p_match_reward
        
        if epoch == config['num_epoch'] - 1:
            end_time = time()
            print('Epoch {} starts !'.format(epoch))
            print('-' * 80)
            
            file_basename = 'runs/n{}m{}d{}ka{}eta{}eps{}iter{}alg{}trial{}regret_'.format(config['points_per_iter'], 
            config['feature_dim'], config['label_dim'],
            config['KA_norm'],config['eta'],config['epsilon'],config['learn_iter'],
            config['learning_alg'],trial_idx)
            
            for algorithm, dataset in zip(['p2p','bandit','p2p_o','p2p_b'], 
                                          [regret_p2p,regret_bandit,regret_p2p_o,regret_p2p_b]):
                file_name = file_basename + algorithm + ".npy"
                np.save(file_name,dataset)

nbc_palette = ['#e16428', '#b42846', '#008cc3', '#00a846']
sns.palplot(sns.color_palette(nbc_palette))

file_name = 'figures/n{}m{}d{}ka{}eta{}eps{}iter{}alg{}FINAL{}.png'.format(config['points_per_iter'], config['feature_dim'], config['label_dim'],
config['KA_norm'],config['eta'],config['epsilon'],config['learn_iter'],config['learning_alg'],config['num_trial'])
data = [regret_p2p,regret_bandit,regret_p2p_o,regret_p2p_b]
labels = ['PROOF', 'Vanilla OFU', 'PROOF (Optimization)', 'PROOF (Bandit)']
plot_data(data,labels,file_name)

file_name = 'figures/n{}m{}d{}ka{}eta{}eps{}iter{}alg{}FINAL{}_match.png'.format(config['points_per_iter'], config['feature_dim'], config['label_dim'],
config['KA_norm'],config['eta'],config['epsilon'],config['learn_iter'],config['learning_alg'],config['num_trial'])
data = [regret_match_p2p,regret_match_bandit]
labels = ['PROOF', 'Vanilla OFU']
plot_data(data,labels,file_name)
