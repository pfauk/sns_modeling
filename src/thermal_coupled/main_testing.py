"""
Script file for testing larger problem sizes 

Two main problem instances:
-Hydrocarbon mixture of primairly aromatics
-Linear alkane mixture

"""

import sys
import os
import logging
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition
from pyomo.util.model_size import build_model_size_report
from pyomo.util.infeasible import (
    log_infeasible_constraints,
    log_infeasible_bounds,
    find_infeasible_constraints,
)
from utils import (
    data,
    pprint_network,
    pprint_tasks,
    save_model_to_file,
    save_solution_to_file)
from superstructure.stn import stn
from thermal_coupled.therm_dist import build_model

logging.basicConfig(level=logging.INFO)

# specify number of components and data file name
n = 6
data_file_name = '6_comp_alkanes.xlsx'

# import problem data for system and relevant species to data object
hydrocarbon_data = data(data_file_name)

# build state-task network superstrucutre and associated index sets
network_superstructure = stn(n)
network_superstructure.generate_tree()
network_superstructure.generate_index_sets()

model = build_model(network_superstructure, hydrocarbon_data)

# uncomment below line to save the Pyomo model to a txt file to examine model
# save_model_to_file(model, '4_comp_pyomo_model_alkanes')

mdl_size = build_model_size_report(model)

# SOLUTION OPTION 1) Transform GDP to MINLP and solve with BARON
# =============================================================

pyo.TransformationFactory('core.logical_to_linear').apply_to(model)

# applying Big-M transformation
reformulation = pyo.TransformationFactory('gdp.hull')

reformulation.apply_to(model)

# BARON is accessed through GAMS to solve the MINLP
io_options=dict(resLim=38000)
solver = pyo.SolverFactory('gams:baron', io_options=io_options)

result = solver.solve(model, tee=True)

# SOLUTION OPTION 2) Solution of GDP with Logic Based approach using GDPopt
# =======================================================================
# result = pyo.SolverFactory('gdpopt.loa').solve(model, nlp_solver="gams:ipopt", mip_solver="gurobi", tee=True)

# logging infeasible constraints if any
if result.solver.termination_condition == TerminationCondition.infeasible:
    print()
    log_infeasible_constraints(model)
    log_infeasible_bounds(model)
    infeasible_cosntraints = find_infeasible_constraints(model)

    for constraint in infeasible_cosntraints:
        print(constraint[0])
else:
    print("Model solved successfully")


# SOLUTION OUTPUT
# =================================================================
pprint_network(model)

print()
print('Model Size')
print(mdl_size)

print()
print('STN Size')
print(f'Number of STATES in Network: {len(network_superstructure.STATES)}')
print(f'Number of TASKS in Network: {len(network_superstructure.TASKS)}')

print()
print('Underwood Roots')
print("================================================")
for t in model.TASKS:
    for r in model.r:
        print(f'{t} {r} Root value: {pyo.value(model.rud[(t, r)])}')

model.r.pprint()
model.RUA.pprint()

# # uncomment below line to save the solution output to a txt file
# save_solution_to_file(model, '5_comp_solution_test')
