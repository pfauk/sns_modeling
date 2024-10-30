"""Main script file to build and solve models for thermally coupled distillation columns

User specifies the number of components in the mixture and the file name for the data sheet
to import from

"""

import logging
import os
import pyomo.environ as pyo
from pyomo.util.infeasible import log_infeasible_constraints, find_infeasible_constraints
from pyomo.util.model_size import build_model_size_report
from idaes.core.util.model_statistics import report_statistics
from utils import (
    Data,
    get_model_type,
    pprint_network,
    pprint_tasks,
    save_model_to_file,
    save_solution_to_file,
    get_model_type,
    print_constraint_type,
    recover_original)
from superstructure.stn import stn
from superstructure.stn_nonconsecutive import stn_nonconsecutive
from thermal_coupled.therm_dist import build_model, solve_model


# specify number of components and data file name
n = 4

data_file_name = os.path.join('notebook_examples', '4_comp_linear_hydrocarbons.xlsx')

# data_file_name = '6_comp_alkanes.xlsx'

# import problem data for system and relevant species to data object
mixture_data = Data(data_file_name)

# build state-task network superstrucutre and associated index sets
network_superstructure = stn(n)
network_superstructure.generate_tree()
network_superstructure.generate_index_sets()

# function call returns the Pyomo model object and a dictionary of scaling factors for the cost coefficients
model, scaling_factors = build_model(network_superstructure, mixture_data, scale=True)

print()
print('Inlet data')
print('================================================================')
print(mixture_data.system_df)

print()
print('Mixture species data')
print('================================================================')
print(mixture_data.species_df)

print()
print(f'Model type before transformation: {get_model_type(model)}')


# development of heuristic solution
solved_model, results = solve_model(model)

pprint_network(model)

# uncomment below line to save the solution output to a txt file
# save_solution_to_file(model, '4_comp_solution_2')

# save_model_to_file(model, '3_comp_pyomo_model_solution')
