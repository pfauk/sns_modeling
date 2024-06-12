"""Main script file to build and solve models for thermally coupled distillation columns"""
import sys
from math import pi
import random
import numpy as np
import idaes
import pyomo.environ as pyo
from pyomo.util.infeasible import log_infeasible_constraints, log_infeasible_bounds, find_infeasible_constraints
from pyomo.gdp import Disjunction, Disjunct
from utils import (
    data,
    pprint_network,
    pprint_network_to_file,
    pprint_column, output_model,
    IntHeatExchanger, FinalHeatExchanger)
from superstructure.stn import State, Task, stn
from thermal_coupled.therm_dist import build_model

n = 3  # specify number of components

problem_data = data('3_comp.xlsx')

# build state-task network superstrucutre and associated index sets
network_superstructure = stn(n)
network_superstructure.generate_tree()
network_superstructure.generate_index_sets()
network_superstructure.print_tree()

model = build_model(network_superstructure)

print(type(problem_data))

# SOLUTION
# ================================================
pyo.TransformationFactory('core.logical_to_linear').apply_to(model)

# applying Big-M transformation
mbigm = pyo.TransformationFactory('gdp.bigm')

mbigm.apply_to(model)

solver = pyo.SolverFactory('gams:baron')
status = solver.solve(model, tee=True)

# # =================================================================
# # solution of GDP with L-bOA
# results = pyo.SolverFactory('gdpopt.loa').solve(m, nlp_solver='ipopt', mip_solver='gams:cplex', tee=True)

# # =================================================================
# pprint_network(m)

# print()
# print('INTERMEDIATE HEAT EXCHANGERS')
# for i in m.ISTATE:
#     print(f'{i}: {pyo.value(m.int_heat_exchanger[i].indicator_var)}')


# print()
# print('FINAL HEAT EXCHANGERS')
# for i in m.COMP:
#     print(f'{i}: {pyo.value(m.final_heat_exchanger[i].indicator_var)}')

# print()
# print('Active Tasks')
# for t in m.TASKS:
#     print(f'{t}: {pyo.value(m.column[t].indicator_var)}')

# for t in m.TASKS:
#     print(f'Qreb {t}: {pyo.value(m.Qreb[t])}')
#     print(f'Qcond {t}: {pyo.value(m.Qcond[t])}')
#    # print(f'Heat Exchanger cost: $ {pyo.value(m.final_heat_exchanger_cost[t])}')


# # uncomment the line below if you want to output the Pyomo model to a text file
# output_model(m, '3_Component_GDP_Model')

# #pprint_network_to_file(m, 'temp_file')


# print()
# print('Exchanger area')
# for i in m.COMP:
#     print(f'{i} {pyo.value(m.area_final_exchanger[i])}')

# print()
# print('Exchanger cost')
# for i in m.COMP:
#     print(f'{i} {pyo.value(m.final_reboiler_cost[i])}')
#     print(f'{i} {pyo.value(m.final_condenser_cost[i])}')
