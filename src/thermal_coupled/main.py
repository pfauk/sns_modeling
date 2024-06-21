import sys
import os
import numpy as np
import logging
import pyomo.environ as pyo
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

"""Main script file to build and solve models for thermally coupled distillation columns

User specifies the number of components in the mixture and the file name for the data sheet
to import from

"""

# specify number of components and data file name
n = 4
data_file_name = '4_comp.xlsx'

# import problem data for system and relevant species to data object
hydrocarbon_data = data(data_file_name)

# build state-task network superstrucutre and associated index sets
network_superstructure = stn(n)
network_superstructure.generate_tree()
network_superstructure.generate_index_sets()

model = build_model(network_superstructure, hydrocarbon_data)

# # uncomment below line to save the Pyomo model to a txt file to examine model
save_model_to_file(model, '4_comp_pyomo_model')

# SOLUTION
# ================================================
pyo.TransformationFactory('core.logical_to_linear').apply_to(model)

# applying Big-M transformation
mbigm = pyo.TransformationFactory('gdp.bigm')

mbigm.apply_to(model)

solver = pyo.SolverFactory('gams:baron')
results = solver.solve(model, tee=True)

# uncomment below line if you want to see solver results, problem size
# print(results)

# Log infeasible constraints if any
logging.basicConfig(level=logging.INFO)
log_infeasible_constraints(model)

# =================================================================
# solution of GDP with L-bOA
# results = pyo.SolverFactory('gdpopt.loa').solve(m, nlp_solver='ipopt', mip_solver='gams:cplex', tee=True)

# SOLUTION OUTPUT
# =================================================================
# pprint_network(model)

# uncomment below line to save the solution output to a txt file
save_solution_to_file(model, '4_comp_solution')
