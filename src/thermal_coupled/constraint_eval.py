"""
Script file for examining constraint reformulation to an MIQCP

"""

import sys
import os
import logging
import pyomo.environ as pyo
from pyomo.core.expr.visitor import polynomial_degree
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
from utils import get_model_type, print_constraint_type

logging.basicConfig(level=logging.INFO)

# specify number of components and data file name
n = 3
data_file_name = '3_comp_alkanes.xlsx'

# import problem data for system and relevant species to data object
hydrocarbon_data = data(data_file_name)

# build state-task network superstrucutre and associated index sets
network_superstructure = stn(n)
network_superstructure.generate_tree()
network_superstructure.generate_index_sets()

model = build_model(network_superstructure, hydrocarbon_data)

# applying Big-M transformation
tansform = pyo.TransformationFactory('gdp.bigm')

tansform.apply_to(model)

mdl_size = build_model_size_report(model)

mdl_type = get_model_type(model)

print(f'Model type: {mdl_type}')

# print_constraint_type(model)

# print_constraint_type(model)

# solver_options = {
#     'TimeLimit': 20000,
#     'NonConvex': 2,
#     'NumericFocus': 1
# }

# solver = pyo.SolverFactory('gurobi')
# results = solver.solve(model, options=solver_options, tee=True)
