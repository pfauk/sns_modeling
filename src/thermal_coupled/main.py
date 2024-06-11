"""Main script file to build and solve models for thermally coupled distillation columns"""

from superstructure.stn import State, Task, stn


n = 3  # specify number of components

# build state-task network superstrucutre and associated index sets
network_superstructure = stn(n)
network_superstructure.generate_tree()
network_superstructure.generate_index_sets()

