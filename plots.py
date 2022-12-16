from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import numpy as np

def animate_function_process(i,c,action,mask):
    plt.clf()

    x = [i/2 for i in range(len(c[0]))]

    final_labels = list(c[i//10]*action[i//10]*mask[i//10])
    
    y_labels = ["Available","Notified","Good match","Final Result"][::-1]
    for j,dataset in enumerate([c[i//10],action[i//10],mask[i//10],final_labels][::-1]):
        y_pos = [j for k in range(len(dataset))]
        colors = ['r' if dataset[k] == 0 else 'b' for k in range(len(dataset))]
        plt.scatter(x,y_pos,c=colors,s=200)
        plt.text(x[0]-0.6,y_pos[0]-0.05,y_labels[j])

    plt.xlim(-0.75,max(x)+0.2)
    plt.title("Notification algorithm for task {}".format(i//10+1))


def animation_function_matches(i,source_locations,destination_locations,matches,c,action,mask):
    print("Calling with i {}".format(i))
    if i<200:
        if i%10 == 0:
            animate_function_process(i,c,action,mask)
        return 

    i = i-200
    
    plt.clf()
    plt.scatter([j[0] for j in source_locations], [j[1] for j in source_locations],s=100,c='b')
    plt.scatter([j[0] for j in destination_locations], [j[1] for j in destination_locations],s=100,c='r')

    all_match_points = []

    for match in matches:
        start = np.array(source_locations[match[0]])
        end = np.array(destination_locations[match[1]])

        current_point = start + min(i/50,1) * (end-start)
        all_match_points.append(current_point)

    plt.scatter([j[0] for j in all_match_points], [j[1] for j in all_match_points],s=10,c='black')


def animate_matches(source_locations,destination_locations,matches,c,action,mask):
    fig = plt.figure(figsize=(7,5))
    num_epochs = 300
    animation = FuncAnimation(fig, lambda s: animation_function_matches(s,
        source_locations, destination_locations,matches, c,action,mask), 
                              interval = num_epochs, frames=num_epochs)
    
    animation.save('figures/animation.gif', writer='imagemagick', fps=10)


def make_movie(env,action,matches,c):
    source_locations = env.source_locations
    destination_locations = env.destination_locations
    mask = env._m
    
    # Matches are between tasks and people
    # So turn this into start-end 
    new_matches = []
    for i,j in matches:
        new_matches.append((env.data_sources[i],env.data_destinations[i]))
    
    animate_matches(source_locations,destination_locations,new_matches,c,action,mask)