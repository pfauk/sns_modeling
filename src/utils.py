import sys
import pyomo.environ as pyo
import numpy as np

def output_model(mdl, file_name):
    """Function outputs the Pyomo model to a text file in the current directory
    Args:
    -mdl: Pyomo model
    -file_name: string for desired file name

    Returns:
    -None: creates .txt file
    """
    # use utf-8 encoding, standard Windows cp1252 does not support logical characters 
    with open(file_name + '.txt', 'w', encoding='utf-8') as f:
        # Redirect stdout to file
        sys.stdout = f
        mdl.pprint()

    # Reset stdout to its default value
    sys.stdout = sys.__stdout__

    return None

class Column:
    """Class to encapuslate data for a given column (separation task) for easier display"""

    def __init__(self, mdl, k):
        """Initialization takes a Pyomo model and the index, k, of the column"""
        # initialization of Boolean varibale for if the column is active
        self.active = pyo.value(mdl.column[k].indicator_var)
        self.col_index = str(k)
        self.obj = pyo.value(mdl.obj())
        self.cost = pyo.value(mdl.column_cost[k])

        # initialization of total flows
        self.FT = pyo.value(mdl.FT[k])
        self.DT = pyo.value(mdl.DT[k])
        self.BT = pyo.value(mdl.BT[k])

        # initialization of component flows
        self.F = {}
        self.D = {}
        self.B = {}

        for i in mdl.COMP:
            self.F[('F', i)] = pyo.value(mdl.F[(i, k)])
            self.D[('D', i)] = pyo.value(mdl.D[(i, k)])
            self.B[('B', i)] = pyo.value(mdl.B[(i, k)])

        # initialization of internal coumn flows
        self.Vr = pyo.value(mdl.Vr[k])
        self.Lr = pyo.value(mdl.Lr[k])
        self.Ls = pyo.value(mdl.Ls[k])
        self.Vs = pyo.value(mdl.Vs[k])

        # initialization of column physical dimensions
        self.area = pyo.value(mdl.Area[k])  # column area [m^2]
        self.diameter = 2 * np.sqrt(self.area / np.pi)  # column diameter [m]
        self.height = pyo.value(mdl.height[k])  # column height [m]
        self.trays = pyo.value(mdl.Ntray[k])  # number of trays

        # initialization of heat exchanger (reboiler and condenser)
        self.Qreb = pyo.value(mdl.Qreb[k])
        self.Qcond = pyo.value(mdl.Qcond[k])

    def display_simple_col(self):
        # display just the value for total molar flows of feed, distillate, and bottoms
        if self.active is False:
            print(f'Separation task {self.col_index} is not active')
        elif self.active is True:
            print()
            print(f'Column {self.col_index}:')
            print(f'Column cost ${self.cost:,.2f}')
            print(f'Number of trays: {self.trays:1f}')
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
    print(f'System Feed Total: {feed_index} ({feed_val} [kmol/hr])')
    for i in mdl.COMP:
        print(f'Feed {i} {mdl.F0_comp[i]}')

    # build up a dictionary with Column objects of active columns
    active_columns = {}
    for k in mdl.TASKS:
        temp = pyo.value(mdl.column[k].indicator_var)
        if temp is True:
            active_columns[k] = Column(mdl, k)

    print()
    print('Active Columns')
    print("================================")
    for cols in active_columns.keys():
        print(active_columns[cols].col_index)

    # do a simple display (total flows) for every active column
    for i in active_columns.keys():
        active_columns[i].display_simple_col()

    print()
    print("================================")
    print('DETAILED NETWORK VIEW')
    print("================================")
    print()

    for i in active_columns.keys():
        active_columns[i].display_complete_col()
