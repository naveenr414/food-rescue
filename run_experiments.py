import os
from copy import deepcopy

default_dictionary = {'alias': 'mlp_factor8neg4_bz256_166432168_pretrain_reg_0.0000001',
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

config = deepcopy(default_dictionary)

def write_dictionary_and_run():
    w = open("config.txt","w")
    w.write(str(config))
    w.close()
    
    os.system("python train.py")

# Run the default experiment
print("Running the default experiment")
write_dictionary_and_run()
    
# Number of agent experiments
for num_agents in [3,10,20]:
    print("Running with num agents {}".format(num_agents))
    config['label_dim'] = num_agents
    write_dictionary_and_run()

# Points per iteration experiments
config = deepcopy(default_dictionary)
for points_per_iter in [5,10,40]:
    print("Running with points_per_iter {}".format(points_per_iter))
    config['points_per_iter'] = points_per_iter
    write_dictionary_and_run()

# City size experiments 
config = deepcopy(default_dictionary)
for num_sources in [10,20]:
    print("Running with num_sources {}".format(num_sources))
    config['num_sources'] = num_sources
    config['num_destinations'] = num_sources
    write_dictionary_and_run()
    
# Range experiments
config = deepcopy(default_dictionary)
for mean_range in [0.5,0.9]:
    print("Running with mean range {}".format(mean_range))
    config['mean_volunteer_attributes'][1] = mean_range
    write_dictionary_and_run()
    
# Capacity experiments
config = deepcopy(default_dictionary)
for mean_capacity, trip_capacity in [(0.6,0.7),(0.5,0.8),(0.8,0.5),(0.9,0.3),(0.3,0.9)]:
    print("Running with mean capacity {} and trip capacity {}".format(mean_capacity,trip_capacity))
    config['mean_volunteer_attributes'][0] = mean_capacity
    config['mean_task_attributes'][0] = trip_capacity
    write_dictionary_and_run()
    
print("Finished running!")
