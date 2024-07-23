"""Main script file to build and solve models for thermally coupled distillation columns

User specifies the number of components in the mixture and the file name for the data sheet
to import from

"""

import logging
import pyomo.environ as pyo
from pyomo.util.infeasible import log_infeasible_constraints
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
from thermal_coupled.therm_dist import build_model


# specify number of components and data file name
n = 3
data_file_name = '3_comp_test.xlsx'

# import problem data for system and relevant species to data object
hydrocarbon_data = Data(data_file_name)

# build state-task network superstrucutre and associated index sets
network_superstructure = stn(n)
network_superstructure.generate_tree()
network_superstructure.generate_index_sets()


print(hydrocarbon_data.root_lower_bounds)


model = build_model(network_superstructure, hydrocarbon_data)

print()
print('Model size before transformation:')
print(build_model_size_report(model))

# # # uncomment below line to save the Pyomo model to a txt file to examine model
# save_model_to_file(model, '3_comp_pyomo_model')

# # SOLUTION
# # ================================================
pyo.TransformationFactory('core.logical_to_linear').apply_to(model)

# applying Big-M transformation
mbigm = pyo.TransformationFactory('gdp.bigm')

mbigm.apply_to(model)

# MODEL ANALYSIS
# =================================================================
print(report_statistics(model))

print('Model size after transformation:')
print(build_model_size_report(model))

print()
print(f'Model type after transformation: {get_model_type(model)}')


solver = pyo.SolverFactory('gurobi')

# Gurobi solver options
solver.options = {'NumericFocus': 2}

results = solver.solve(model, tee=True)

# uncomment below line if you want to see solver results, problem size
# print(results)

# Log infeasible constraints if any
logging.basicConfig(level=logging.INFO)
log_infeasible_constraints(model)


# SOLUTION OUTPUT
# =================================================================
pprint_network(model)

# uncomment below line to save the solution output to a txt file
# save_solution_to_file(model, '3_comp_solution')

# check Underwood roots
print()
print('Underwood Roots')
for t in model.TASKS:
    if pyo.value(model.column[t].indicator_var):
        for r in model.roots:
            print(f'  Phi({t}, {r}): {pyo.value(model.rud[(t, r)])}')

# check z values
print()
print('Z values')
for t in model.TASKS:
    if pyo.value(model.column[t].indicator_var):
        for r in model.roots:
            for i in model.COMP:
                print(f'  z({i}, {r}): {pyo.value(model.z[(i, t, r)])}')
