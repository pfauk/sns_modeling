import sys
import os
import math
from math import pi
import pandas as pd
import pyomo.environ as pyo
from pyomo.core.expr.visitor import polynomial_degree
import numpy as np
from typing import Tuple, Dict

class Column:
    """Class to encapuslate data for a given column (separation task) for easier display.
    Columns are indexed by the set m.TASKS, which represents all possible separation tasks
    with a separation network."""

    def __init__(self, mdl, t):
        """Initialization takes a Pyomo model and the index, k, of the column"""
        # initialization of Boolean varibale for if the column is active
        self.active = pyo.value(mdl.column[t].indicator_var)
        self.col_index = str(t)
        self.obj = pyo.value(mdl.obj())
        self.cost = pyo.value(mdl.column_cost[t])

        # initialization of total flows
        self.FT = pyo.value(mdl.FT[t])
        self.DT = pyo.value(mdl.DT[t])
        self.BT = pyo.value(mdl.BT[t])

        # initialization of component flows
        self.F = {}
        self.D = {}
        self.B = {}

        for i in mdl.COMP:
            self.F[('F', i)] = pyo.value(mdl.F[(i, t)])
            self.D[('D', i)] = pyo.value(mdl.D[(i, t)])
            self.B[('B', i)] = pyo.value(mdl.B[(i, t)])

        # initialization of internal column flows
        self.Vr = pyo.value(mdl.Vr[t])
        self.Lr = pyo.value(mdl.Lr[t])
        self.Ls = pyo.value(mdl.Ls[t])
        self.Vs = pyo.value(mdl.Vs[t])

        # initialization of column physical dimensions
        self.area = pyo.value(mdl.Area[t])  # column area [m^2]
        self.diameter = 2 * pyo.sqrt(self.area / pi)  # column diameter [m]
        self.height = pyo.value(mdl.height[t])  # column height [m]
        self.trays = math.ceil(pyo.value(mdl.Ntray[t]))  # number of trays

        # initialization of heat exchanger (reboiler and condenser)
        self.Qreb = pyo.value(mdl.Qreb[t])
        self.Qcond = pyo.value(mdl.Qcond[t])

    def display_simple_col(self):
        # display just the value for total molar flows of feed, distillate, and bottoms
        if self.active is False:
            print(f'Separation task {self.col_index} is not active')
        elif self.active is True:
            print()
            print(f'Column {self.col_index}:')
            print(f'Column cost ${self.cost:,.2f}')
            print(f'Number of trays: {self.trays:,.1f}')
            print(f'Column height: {self.height:1.1f} [m]')
            print(f'Column diameter: {self.diameter:1.1f} [m^2]')
            print('================================')
            print(f'Feed flow (FT): {self.FT:1.4f}')
            print(f'Distillate (DT): {self.DT:1.4f}')
            print(f'Bottoms (BT): {self.BT:1.4f}')

    def display_complete_col(self):
        # display all total, component, and internal flows of the column
        print()
        print('================================')
        if self.active is False:
            print(f'Separation task {self.col_index} is not active')
        elif self.active is True:
            print(f'Separation task {self.col_index} is active')
            print()
            print(f'Total Flows for Column {self.col_index} [kmol/hr]:')
            print(f'Feed flow (FT): {self.FT:1.4f}')
            print(f'Distillate (DT): {self.DT:1.4f}')
            print(f'Bottoms (BT): {self.BT:1.4f}')

            print()
            print('================================')
            print(f'Component Flows for Column {self.col_index}')

            print('Feed Component Flows:')
            for i in self.F.keys():
                print(f'Feed ({i[1]}): {self.F[i]:1.4f}')

            print()
            print('Distillate Component Flows:')
            for j in self.D.keys():
                print(f'Distillate ({j[1]}): {self.D[j]:1.4f}')

            print()
            print('Bottoms Component Flows:')
            for k in self.B.keys():
                print(f'Bottoms ({k[1]}): {self.B[k]:1.4f}')

            print()
            print('Internal Flows of Column')
            print(f'Vr {self.Vr:1.4}')
            print(f'Lr {self.Lr:1.4}')
            print(f'Vs {self.Vs:1.4}')
            print(f'Ls {self.Ls:1.4}')


class IntHeatExchanger:
    """Class to encapuslate data for intermediate heat exchanger for easier display.
    Intermediate heat exchangers are indexed by the set m.ISTATE, which is all intermediate states in
    a separation network"""

    def __init__(self, mdl, s):
        self.active = pyo.value(mdl.int_heat_exchanger[s].indicator_var)
        self.exchanger_index = str(s)
        self.cost = 0  # initialize to 0 and check if exchanger is a condenser or reboiler
        self.exchanger_area = 0
        self.heat_duty = 0

        # get lists of active rectifying and stripping tasks that produce this intermediate state
        # should not be more than 2 tasks, one from a rectifying section and one from a stripping section
        active_strip_tasks = []
        active_rec_tasks = []

        if s in mdl.IREC_m:
            rec_tasks = list(mdl.IREC_m[s])

            for t in rec_tasks:
                if pyo.value(mdl.column[t].indicator_var):
                    active_rec_tasks.append(t)

            self.active_rec_tasks = active_rec_tasks
        else:
            self.active_rec_tasks = None

        if s in mdl.ISTRIP_m:
            strip_tasks = list(mdl.ISTRIP_m[s])

            for t in strip_tasks:
                if pyo.value(mdl.column[t].indicator_var):
                    active_strip_tasks.append(t)

            self.active_strip_tasks = active_strip_tasks
        else:
            self.active_strip_tasks = None

        # if the heat exchanger is active, determine if it is a reboiler or condenser
        # based on what type of separation tasks are producting it
        if self.active:
            if self.active_rec_tasks and not self.active_strip_tasks:
                self.is_condenser = True
                self.cost = mdl.inter_condenser_cost[s]
                self.exchanger_area = pyo.value(mdl.area_intermediate_reboiler[s]) + pyo.value(mdl.area_intermediate_condenser[s])
                condenser_duty = pyo.value(sum(mdl.Qcond[t] for t in active_rec_tasks))
            else:
                self.is_condenser = False
                condenser_duty = 0

            if self.active_strip_tasks and not self.active_rec_tasks:
                self.is_reboiler = True
                self.cost = mdl.inter_reboiler_cost[s]
                self.exchanger_area = pyo.value(mdl.area_intermediate_reboiler[s]) + pyo.value(mdl.area_intermediate_condenser[s])
                reboiler_duty = pyo.value(sum(mdl.Qreb[t] for t in active_strip_tasks))
            else:
                self.is_reboiler = False
                reboiler_duty = 0

            self.heat_duty = reboiler_duty + condenser_duty

        elif not self.active:
            self.is_condenser = False
            self.is_reboiler = False

    def display_exchanger(self):
        # display costs, area, and heat duty of heat exchanger
        print()
        if self.active is False:
            print(f'Heat Exchanger for final product {self.exchanger_index} is not active')
        elif self.active is True:
            if self.is_reboiler is True:
                print(f'State {self.exchanger_index} has associated reboiler')
                print(f'Capital cost of reboiler: ${pyo.value(self.cost):,.2f}')
                print(f'Exchanger area: {self.exchanger_area:,.2f} [m^2]')
                print(f'Heat Duty: {self.heat_duty:,.1f} [10^3 kJ/hr]')
            if self.is_condenser is True:
                print(f'State {self.exchanger_index} has associated condenser')
                print(f'Capital cost of condenser: ${self.cost:,.2f}')
                print(f'Exchanger area: {self.exchanger_area:,.2f} [m^2]')
                print(f'Heat Duty: {self.heat_duty:,.1f} [10^3 kJ/hr]')

class FinalHeatExchanger:
    """Class to encapuslate data for final product heat exchanger for easier display.
    Final product heat exchangers are indexed by the set m.COMP, which is all the pure components in a
    separation network"""

    def __init__(self, mdl, i):
        self.active = pyo.value(mdl.final_heat_exchanger[i].indicator_var)
        self.exchanger_index = str(i)
        self.cost = 0  # initialize to 0 and check if exchanger is a condenser or reboiler
        self.exchanger_area = 0
        self.heat_duty = 0

        # get lists of active rectifying and stripping tasks that produce this final state
        # should not be more than 2 tasks, one from a rectifying section and one from a stripping section
        active_rec_tasks = []
        active_strip_tasks = []

        if i in mdl.PRE_i:
            rec_tasks = list(mdl.PRE_i[i])

            for t in rec_tasks:
                if pyo.value(mdl.column[t].indicator_var):
                    active_rec_tasks.append(t)

            self.active_rec_tasks = active_rec_tasks
        else:
            self.active_rec_tasks = None

        if i in mdl.PST_i:
            strip_tasks = list(mdl.PST_i[i])

            for t in strip_tasks:
                if pyo.value(mdl.column[t].indicator_var):
                    active_strip_tasks.append(t)

            self.active_strip_tasks = active_strip_tasks
        else:
            self.active_strip_tasks = None

        # if the heat exchanger is active, determine if it is a reboiler or condenser
        # based on what type of separation tasks are producting it
        if self.active:
            if self.active_rec_tasks and not self.active_strip_tasks:
                self.is_condenser = True
                self.cost = pyo.value(mdl.final_condenser_cost[i])
                self.exchanger_area = pyo.value(mdl.area_final_reboiler[i]) + pyo.value(mdl.area_final_condenser[i])
                condenser_duty = pyo.value(sum(mdl.Qcond[t] for t in active_rec_tasks))
            else:
                self.is_condenser = False
                condenser_duty = 0

            if self.active_strip_tasks and not self.active_rec_tasks:
                self.is_reboiler = True
                self.cost = pyo.value(mdl.final_reboiler_cost[i])
                self.exchanger_area = pyo.value(mdl.area_final_reboiler[i]) + pyo.value(mdl.area_final_condenser[i])
                reboiler_duty = pyo.value(sum(mdl.Qreb[t] for t in active_strip_tasks))
            else:
                self.is_reboiler = False
                reboiler_duty = 0

            self.heat_duty = reboiler_duty + condenser_duty

        elif not self.active:
            self.is_condenser = False
            self.is_reboiler = False

    def display_exchanger(self):
        # display costs, area, and heat duty of heat exchanger
        print()
        if self.active is False:
            print(f'Heat Exchanger for final product {self.exchanger_index} is not active')
        elif self.active is True:
            if self.is_reboiler is True:
                print(f'Final Prodcut {self.exchanger_index} has associated reboiler')
                print(f'Capital cost of reboiler: ${self.cost:,.2f}')
                print(f'Exchanger area: {self.exchanger_area:,.2f} [m^2]')
                print(f'Heat Duty: {self.heat_duty:,.1f} [10^3 kJ/hr]')
            if self.is_condenser is True:
                print(f'Final Prodcut {self.exchanger_index} has associated condenser')
                print(f'Capital cost of condenser: ${self.cost:,.2f}')
                print(f'Exchanger area: {self.exchanger_area:,.2f} [m^2]')
                print(f'Heat Duty: {self.heat_duty:,.1f} [10^3 kJ/hr]')


def pprint_column(mdl, k):
    # general function that can use the column object to display the column
    column = Column(mdl, k)
    column.display_complete_col()

def pprint_network(mdl):
    """Function takes an input of a distillation network modeled as a pyomo object and outputs the solution
    Args:
        -mdl (pyomo.ConcreteModel): pyomo model object

    Returns:
        -None: function does visual display, does not return a value
    """

    # get index and value for feed
    element = next(iter(mdl.Feed))
    feed_index = str(pyo.value(element))

    feed_val = pyo.value(mdl.F0)

    print()
    print("================================")
    print('NETWORK OVERVIEW')
    print("================================")
    print(mdl.name)
    print('*All Flow Units in [kmol/hr]*')
    print()
    print(f'System Objective: ${mdl.obj():,.2f}')
    print()
    print(f'CAPEX: ${pyo.value(mdl.CAPEX):,.2f}')
    print(f'OPEX: ${pyo.value(mdl.OPEX):,.2f}')
    print()
    print(f'System Feed Total: {feed_index} ({feed_val} [kmol/hr])')
    for t in mdl.COMP:
        print(f'Feed {t} {mdl.F0_comp[t]}')

    # build up a dictionary with Column objects of active columns
    active_columns = {}
    for t in mdl.TASKS:
        temp = pyo.value(mdl.column[t].indicator_var)
        if temp is True:
            active_columns[t] = Column(mdl, t)

    # build up a dictionary with FinalHeatExchanger objects of active heat exchangers associated with final products
    active_final_heat_exchanger = {}
    for i in mdl.COMP:
        temp = pyo.value(mdl.final_heat_exchanger[i].indicator_var)
        if temp is True:
            active_final_heat_exchanger[i] = FinalHeatExchanger(mdl, i)

    # build up a dictionary with IntHeatExchanger objects of active heat exchangers associated with intermediate products
    active_inter_heat_exchanger = {}
    for s in mdl.ISTATE:
        temp = pyo.value(mdl.int_heat_exchanger[s].indicator_var)
        if temp is True:
            active_inter_heat_exchanger[s] = IntHeatExchanger(mdl, s)

    print()
    print('Active Separation Tasks')
    print("================================")
    for cols in active_columns.keys():
        print(active_columns[cols].col_index)

    # do a simple display (total flows) for every active column
    for t in active_columns.keys():
        active_columns[t].display_simple_col()

    print()
    print("================================")
    print('DETAILED NETWORK VIEW')
    print("================================")
    print()

    for t in active_columns.keys():
        active_columns[t].display_complete_col()

    print()
    print("================================")
    print('HEAT EXCHANGERS')
    print("================================")
    print()

    print('Final Product Heat Exchangers')
    for i in active_final_heat_exchanger.keys():
        active_final_heat_exchanger[i].display_exchanger()

    print()
    print('Intermedaite Product Heat Exchangers')
    for s in active_inter_heat_exchanger.keys():
        active_inter_heat_exchanger[s].display_exchanger()

def pprint_tasks(mdl):
    """Function displays all separations tasks and if they are active / inactive"""
    print()
    print('Active Separation Tasks in Network')
    for t in mdl.TASKS:
        print(f'{t}: {pyo.value(mdl.column[t].indicator_var)}')


def save_solution_to_file(mdl, file_name, dir_path=None):
    """
    saves the formatted solution output and solver output to a .txt file

    Args:
    -mdl: Pyomo model
    -results: solver output
    -file_name: string for desired file name
    -dir_path: string for desired directory; default to output to results directory

    Returns:
    -None: creates .txt file
    """

    file_name = str(file_name + '.txt')

    base_dir = os.path.dirname(os.path.realpath(__file__))

    if dir_path:
        directory = os.path.join(base_dir, 'thermal_coupled', dir_path)
    else:
        directory = os.path.join(base_dir, 'thermal_coupled', 'results')

    full_path = os.path.join(directory, file_name)

    # use utf-8 encoding instead of standard Windows cp1252
    with open(full_path, 'w', encoding='utf-8') as f:
        # Redirect stdout to file
        sys.stdout = f
        pprint_network(mdl)

    # Reset stdout to its default value
    sys.stdout = sys.__stdout__

    return None

def save_model_to_file(mdl, file_name, dir_path=None):
    """
    saves the Pyomo model to a .txt file

    Args:
    -mdl: Pyomo model
    -file_name: string for desired file name
    -dir_path: string for desired directory
        default to output to src/thermal_coupled/saved_models directory

    Returns:
    -None: creates .txt file
    """
    file_name = str(file_name + '.txt')

    base_dir = os.path.dirname(os.path.realpath(__file__))

    if dir_path:
        directory = os.path.join(base_dir, 'thermal_coupled', dir_path)
    else:
        directory = os.path.join(base_dir, 'thermal_coupled', 'saved_models')

    full_path = os.path.join(directory, file_name)

    # use utf-8 encoding instead of standard Windows cp1252 for output to handle logical characters
    with open(full_path, 'w', encoding='utf-8') as f:
        # Redirect stdout to file
        sys.stdout = f
        mdl.pprint()

    # Reset stdout to its default value
    sys.stdout = sys.__stdout__

    return None

class Data:
    """
    Class to hold problem data for input to a build_model() function.
    """
    def __init__(self, file_name: str, dir_path: str = None):
        base_dir = os.path.dirname(os.path.realpath(__file__))

        if dir_path:
            data_directory = os.path.join(base_dir, dir_path)
        else:
            data_directory = os.path.join(base_dir, 'data')

        self.filepath = os.path.join(data_directory, file_name)

        self.species_df = pd.read_excel(self.filepath, sheet_name='species')
        self.system_df = pd.read_excel(self.filepath, sheet_name='system')

        # Type cast data to Python floats instead of Numpy float64 in order to work with GDPopt
        self.n = float(self.species_df.shape[0])  # number of components in the system
        self.species_names = dict(zip(self.species_df['index'], self.species_df['Species']))
        self.F0 = float(self.system_df['F0 [kmol/hr]'].iloc[0])
        self.P_abs = float(self.system_df['Pressure [bar]'].iloc[0])
        self.Tf = float(self.system_df['Temp [C]'].iloc[0])

        self.zf = dict(zip(self.species_df['index'], self.species_df['Inlet Mole Frac']))
        self.relative_volatility = dict(zip(self.species_df['index'], self.species_df['Relative Volatility']))
        self.species_densities = dict(zip(self.species_df['index'], self.species_df['Liquid Density [kg/m^3]']))
        self.PM = dict(zip(self.species_df['index'], self.species_df['Molecular Weight']))
        self.Hvap = dict(zip(self.species_df['index'], self.species_df['Enthalpy of Vaporization [kJ/mol]']))
        self.rec = dict(zip(self.species_df['index'], self.species_df['Recovery']))

        # Calculate bounds and initialization for Underwood roots and intermediate variable (z)
        self.root_upper_bounds, self.root_lower_bounds, self.root_initial = self.calculate_root_bounds(self.relative_volatility)
        self.z_upper_bounds, self.z_lower_bounds, self.z_initial = self.calculate_intermediate_bounds(self.relative_volatility)
        
    def calculate_root_bounds(self, alpha:dict) -> Tuple[dict, dict, dict]:
        """Calculate the bounds and initialization for Underwood roots based on relative volatilites

        Args:
            alpha (dict): relative volatilities for n species (e.g. A:10, B:5, C:1,...)

        Returns:
            Tuple[dict]: upper bounds on Underwood roots (dict), lower bounds on Underwood roots (dict),
                         initial values for Underwood roots (dict) as mid point between bounds 
        """
        root_upper_bounds = {}
        root_lower_bounds = {}
        root_initial = {}    
        
        n_components = len(alpha)
        alpha_values = list(alpha.values())
        
        for i in range(n_components - 1):
            root_upper_bounds[f'r{i+1}'] = alpha_values[i] - 0.01
            root_lower_bounds[f'r{i+1}'] = alpha_values[i+1] + 0.01
            root_initial[f'r{i+1}'] = (alpha_values[i] + alpha_values[i+1]) / 2
            
        return root_upper_bounds, root_lower_bounds, root_initial
    
    def calculate_intermediate_bounds(self, alpha:dict) -> Tuple[dict, dict, dict]:
        """Calculate the bounds and initialization for intermediate variable (z) used in reformulation of Underwood equations

        Args:
            alpha (dict): relative volatilities for n species (e.g. A:10, B:5, C:1,...)

        Returns:
            Tuple[dict]: upper bounds on intermediate variable (dict), lower bounds on intermediate variable (dict),
                         initial values of intermediate variable (dict) as mid point between bounds 
        """
        z_upper_bounds = {}
        z_lower_bounds = {}
        z_initial = {}
        
        n_components = len(alpha)
        components = list(alpha.keys())
        
        root_upper_bounds, root_lower_bounds, root_initial = self.calculate_root_bounds(alpha)
        
        # calculating bounds for intermediate variable (z) using bounds from Underwood roots
        for i in components:
            for r in range(1, n_components):
                z_upper_bounds[(i, f'r{r}')] = 1 / (alpha[i] - root_upper_bounds[f'r{r}'])
                z_lower_bounds[(i, f'r{r}')] = 1 / (alpha[i] - root_lower_bounds[f'r{r}'])
                z_initial[(i, f'r{r}')] = (z_upper_bounds[(i, f'r{r}')] + z_lower_bounds[(i, f'r{r}')]) / 2
    
        return z_upper_bounds, z_lower_bounds, z_initial

# utility functions to examine model type
def get_var_type(model: pyo.ConcreteModel) -> dict:
    """retruns the domain of each variable in a Pyomo model 

    Args:
        model (pyo.ConcreteModel): Pyomo model object

    Returns:
        dict: dictionary where keys are (variable, index) and values are 
    """
    # diciontary to store variable types in
    variables = {}

    for variable in model.component_objects(pyo.Var, active=True):
        for index in variable:
            variable_type = variable[index].domain.name
            variables[(variable.name, index)] = variable_type

    return variables

def get_constraint_type(model: pyo.ConcreteModel) -> dict:
    """returns the equation type of the constraints in a Pyomo model.
    
    Constraints can be: linear, quadratic, nth degree polynomial, or nonlinear

    Args:
        model (pyo.ConcreteModel): Pyomo model object

    Returns:
        dict: dictionary where keys are (constraint, index) and values are 
    """
    # dictionary to store constraints in
    constraints = {}

    # get degree / type of model constraints
    for constraint in model.component_objects(pyo.Constraint, active=True):
        for index in constraint:
            constraint_degree = polynomial_degree(constraint[index].expr)
            constraint_type = None
            if constraint_degree == 1:
                constraint_type = 'Linear'
            elif constraint_degree == 2:
                constraint_type = 'Quadratic'
            elif isinstance(constraint_degree, int) and constraint_degree > 2:
                constraint_type = f'Degree {constraint_degree} Polynomial'
            elif constraint_degree is None:
                constraint_type = 'Nonlinear'

            constraints[(constraint.name, index)] = constraint_type

    return constraints

def print_constraint_type(model: pyo.ConcreteModel) -> None:
    """displays constraints in model based on type of equation

    Constraints can be: linear, quadratic, nth degree polynomial, or nonlinear
    
    Args:
        model (pyo.ConcreteModel): Pyomo model object
    
    Returns:
        None: prints contraints and returns None
    """

    linear_constraints = {}
    quadratic_constraints = {}
    polynomial_constraints = {}
    nonlinear_constraints = {}

    # get degree / type of model constraints
    for constraint in model.component_objects(pyo.Constraint, active=True):
        for index in constraint:
            constraint_degree = polynomial_degree(constraint[index].expr)
            if constraint_degree == 1:
                linear_constraints[(constraint.name, index)] = constraint[index].expr
            elif constraint_degree == 2:
                quadratic_constraints[(constraint.name, index)] = constraint[index].expr
            elif isinstance(constraint_degree, int) and constraint_degree > 2:
                polynomial_constraints[(constraint.name, index)] = constraint[index].expr
            elif constraint_degree is None:
                nonlinear_constraints[(constraint.name, index)] = constraint[index].expr

    # displaying constraints in model by type
    if not linear_constraints:
        print('No linear constraints in model')
    else:
        print('Linear constraints in model')
        for index, constr in enumerate(linear_constraints.items(), start=1):
            print()
            print(f'{index}: {constr}')

    print()
    if not quadratic_constraints:
        print('No quadratic constraints in model')
    else:
        print('Quadratic constraints in model')
        print()
        for index, constr in enumerate(quadratic_constraints.values(), start=1):
            print()
            print(f'{index}: {constr}')

    print()
    if not polynomial_constraints:
        print('No higher-order polynomial constraints in model')
    else:
        print('Higher-order polynomial constraints in model')
        for index, constr in enumerate(polynomial_constraints.values(), start=1):
            print()
            print(f'{index}: {constr}')

    print()
    if not nonlinear_constraints:
        print('No nonlinear constraints in model')
    else:
        print('Nonlinear constraints in model')
        for index, constr in enumerate(nonlinear_constraints.values(), start=1):
            print()
            print(f'{index}: {constr}')

    return None

def get_objective_type(model: pyo.ConcreteModel) -> str:
    """returns the equation type of the objective function in a Pyomo model.
    
    Objective can be: linear, quadratic, nth degree polynomial, or nonlinear

    Args:
        model (pyo.ConcreteModel): Pyomo model object

    Returns:
        dict: dictionary where keys are (constraint, index) and values are 
    """

    # get degree / type of model objective
    for obj in model.component_objects(pyo.Objective, active=True):
        obj_degree = polynomial_degree(obj.expr)
        obj_type = None
        if obj_degree == 1:
            obj_type = 'Linear'
        elif obj_degree == 2:
            obj_type = ' Quadratic'
        elif isinstance(obj_degree, int) and obj_degree > 2:
            obj_type = f'Degree {obj_degree} Polynomial'
        elif obj_degree is None:
            obj_type = 'Nonlinear'

    return obj_type


def get_model_type(model: pyo.ConcreteModel) -> str:
    """Determine the type of model

    Args:
        model (pyo.ConcreteModel): Pyomo model object

    Returns:
        str: Type of mathemtical program contained in the Pyomo model object
        LP, QP, IP, QP, NLP, MILP, MIQCP, MINLP, GDP
    """

    # domains of discrete variables in a Pyomo model
    integer_vars = {'Integers', 'Binary', 'NegativeIntegers', 'PositiveIntegers', 'NonPositiveIntegers', 'NonNegativeIntegers'}

    var_types = {'Continuous': False, 'Discrete': False}
    constraint_types = {'Linear': False, 'Quadratic': False, 'Nonlinear': False}

    # get dictionaries of the variables, constratinst, and objective and type/domain
    variables = get_var_type(model)
    constraints = get_constraint_type(model)
    objective = get_objective_type(model)

    # check if there are any any continuous or discrete variables
    for value in variables.values():
        if value in integer_vars:
            var_types['Discrete'] = True
        else:
            var_types['Continuous'] = True

    # check if constraints are linear, quadratic, or nonlinear
    for value in constraints.values():
        if value == 'Linear':
            constraint_types['Linear'] = True
        elif value == 'Quadratic':
            constraint_types['Quadratic'] = True
        elif value == 'Nonlinear' or value.startswith('Degree'):
            constraint_types['Nonlinear'] = True

    if objective == 'Nonlinear' or constraint_types['Nonlinear'] is True:
        if var_types['Discrete'] is True:
            model_type = 'MINLP'

    if objective == 'Linear':
        if constraint_types['Linear'] is True and constraint_types['Quadratic'] is False and constraint_types['Nonlinear'] is False:
            if var_types['Continuous'] is True and var_types['Discrete'] is False:
                model_type = 'LP'
            elif var_types['Discrete'] is True and var_types['Continuous'] is True:
                model_type = 'MILP'
            elif var_types['Discrete'] is True and var_types['Continuous'] is False:
                model_type = 'IP'

    if objective == 'Quadratic' or constraint_types['Quadratic'] is True:
        if constraint_types['Nonlinear'] is False:
            if var_types['Continuous'] is True and var_types['Discrete'] is False:
                model_type = 'QP'
            elif var_types['Discrete'] is True:
                model_type = 'MIQCP'

    if var_types['Continuous'] is True and var_types['Discrete'] is False:
        if objective == 'Nonlinear' or constraint_types['Nonlinear'] is True:
            model_type = 'NLP'



    return  model_type



if __name__ == "__main__":
    hydrocarbon_data = Data('3_comp.xlsx')
