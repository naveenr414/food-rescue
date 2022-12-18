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
              'num_trial': 11,
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
regret_mask = np.zeros((config['num_trial'],config['num_epoch']-1))
regret_match_p2p = np.zeros((config['num_trial'], config['num_epoch']-1))
regret_match_bandit = np.zeros((config['num_trial'], config['num_epoch']-1))
regret_match_mask = np.zeros((config['num_trial'],config['num_epoch']-1))
time_p2p = np.zeros((config['num_trial'], config['num_epoch']-1))
time_bandit = np.zeros((config['num_trial'], config['num_epoch']-1))

def run_one_round(engine,env,data_loader,test_feature,test_label,epoch,trial_idx,regret_matrix, match_matrix, best_match, best_reward):
    action, _time = engine.p2p_an_epoch(data_loader, test_feature, epoch_id=epoch)
    matches = engine.get_matches(action,test_label,env._m)
    ro, rb = env.get_reward(action)
    match_reward = env.get_match_reward(matches)
    engine.update_bandit(ro,rb)
    regret_matrix[trial_idx, epoch-1] = (ro.sum()+rb.sum())-best_reward.sum()
    match_matrix[trial_idx, epoch-1] = best_match - match_reward
    
    return ro, rb, match_reward

for trial_idx in range(config['num_trial']):
    env = P2PEnvironment(config, seed=trial_idx * 10)
    engine_p2p = P2PEngine(env, config, pure_bandit=False, predict_mask=False)
    engine_mask = P2PEngine(env, config, pure_bandit=False, predict_mask=True)
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
        
        ro_bandit, rb_bandit, match_bandit = run_one_round(engine_bandit,env,data_loader,test_feature,test_label,epoch,trial_idx, regret_bandit, regret_match_bandit, best_match_reward, best_reward)
        ro_p2p, rb_p2p, match_p2p = run_one_round(engine_p2p,env,data_loader,test_feature,test_label,epoch, trial_idx, regret_p2p, regret_match_p2p, best_match_reward, best_reward)
        ro_mask, rb_mask, match_mask = run_one_round(engine_mask,env,data_loader,test_feature,test_label,epoch, trial_idx, regret_mask, regret_match_mask, best_match_reward, best_reward)

        env.add_to_data_loader()

        if epoch == config['num_epoch'] - 1:
            end_time = time()
            print('Epoch {} starts !'.format(epoch))
            print('-' * 80)
            
            file_basename = 'runs/n{}m{}d{}ka{}eta{}eps{}iter{}alg{}trial{}regret_'.format(config['points_per_iter'], 
            config['feature_dim'], config['label_dim'],
            config['KA_norm'],config['eta'],config['epsilon'],config['learn_iter'],
            config['learning_alg'],trial_idx)
            
            for algorithm, dataset in zip(['p2p','bandit'], 
                                          [regret_p2p,regret_bandit]):
                file_name = file_basename + algorithm + ".npy"
                np.save(file_name,dataset)

nbc_palette = ['#e16428', '#b42846', '#008cc3', '#00a846', "#f0b428"]
sns.palplot(sns.color_palette(nbc_palette))

file_name = 'figures/n{}m{}d{}ka{}eta{}eps{}iter{}alg{}FINAL{}.png'.format(config['points_per_iter'], config['feature_dim'], config['label_dim'],
config['KA_norm'],config['eta'],config['epsilon'],config['learn_iter'],config['learning_alg'],config['num_trial'])
data = [regret_p2p,regret_mask,regret_bandit]
labels = ['PROOF', 'PROOF+Mask','Vanilla OFU']
plot_data(data,labels,file_name)

file_name = 'figures/n{}m{}d{}ka{}eta{}eps{}iter{}alg{}FINAL{}_match.png'.format(config['points_per_iter'], config['feature_dim'], config['label_dim'],
config['KA_norm'],config['eta'],config['epsilon'],config['learn_iter'],config['learning_alg'],config['num_trial'])
data = [regret_match_p2p,regret_match_mask,regret_match_bandit]
labels = ['PROOF', 'PROOF+Mask', 'Vanilla OFU']
plot_data(data,labels,file_name)
