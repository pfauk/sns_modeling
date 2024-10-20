"""Main script file to build and solve models for thermally coupled distillation columns

User specifies the number of components in the mixture and the file name for the data sheet
to import from

"""

import logging
import os
import pyomo.environ as pyo
from pyomo.util.infeasible import log_infeasible_constraints, find_infeasible_constraints
from pyomo.util.model_size import build_model_size_report
from idaes.core.util.model_statistics import  report_statistics
from utils import (
    Data,
    get_model_type,
    pprint_network,
    pprint_tasks,
    save_model_to_file,
    save_solution_to_file,
    get_model_type,
    print_constraint_type)
from superstructure.stn import stn
from superstructure.stn_nonconsecutive import stn_nonconsecutive
from thermal_coupled.therm_dist_scaled import build_model


# specify number of components and data file name
n = 3

data_file_name = os.path.join('poster_problem', '3_comp_linear_hydrocarbons.xlsx')

# import problem data for system and relevant species to data object
mixture_data = Data(data_file_name)

# build state-task network superstrucutre and associated index sets
network_superstructure = stn(n)
network_superstructure.generate_tree()
network_superstructure.generate_index_sets()

# doing stuff with model scaling for solution
model, scaled_model_gdp = build_model(network_superstructure, mixture_data)

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


# SOLUTION
# ================================================
pyo.TransformationFactory('core.logical_to_linear').apply_to(model)
pyo.TransformationFactory('core.logical_to_linear').apply_to(scaled_model_gdp)

# applying Big-M transformation
mbigm = pyo.TransformationFactory('gdp.bigm')

# apply Big-M transformation to both scaled and unscaled models
mbigm.apply_to(model)
mbigm.apply_to(scaled_model_gdp)

print()
print(f'Model type after transformation: {get_model_type(model)}')

# MODEL ANALYSIS
# =================================================================

print()
print('Model size after transformation:')
print(build_model_size_report(model))

# SCALING
# =================================================================

# model.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    
# define scaling factors for variables with larger bounds
# model.scaling_factor[model.final_condenser_cost] = 1e-7
# model.scaling_factor[model.final_reboiler_cost] = 1e-7
# model.scaling_factor[model.inter_condenser_cost] = 1e-7
# model.scaling_factor[model.inter_reboiler_cost] = 1e-7
# model.scaling_factor[model.CAPEX] = 1e-8
# model.scaling_factor[model.OPEX] = 1e-8
    
# # create a new scaled model
# scaled_model = pyo.TransformationFactory('core.scale_model').create_using(model)

# solving model
solver = pyo.SolverFactory('gurobi')

# 'NumericFocus':2,
# Gurobi solver options
solver.options = {'nonConvex': 2,
                  'NumericFocus':2}

# results_unscaled= solver.solve(model, tee=True)
results_scaled = solver.solve(model, tee=True)

# propagate solution of scaled model back to the unscaled model
pyo.TransformationFactory('core.scale_model').propagate_solution(scaled_model_gdp, model)

# Log infeasible constraints if any
logging.basicConfig(level=logging.INFO)
log_infeasible_constraints(model)
find_infeasible_constraints(model)

# SOLUTION OUTPUT
# =================================================================

# solution comparison for both scaled and unscaled models
pprint_network(model)

# uncomment below line to save the solution output to a txt file
# save_solution_to_file(model, '4_comp_solution_2')

# save_model_to_file(model, '3_comp_pyomo_model_solution')

print()

for i in model.COMP:
    print(f'Final heat exchanger active: {i} {pyo.value(model.final_heat_exchanger[i].indicator_var)}')

print()
for s in model.ISTATE:
    print(f'Intermediate heat exchanger active: {s} {pyo.value(model.int_heat_exchanger[s].indicator_var)}')

print()
for k in model.TASKS:
    print(f'Task ({k}): {pyo.value(model.column[k].indicator_var)}')
    