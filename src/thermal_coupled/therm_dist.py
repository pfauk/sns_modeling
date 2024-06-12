"""GDP model of distillation sequences

GDP Model for the optimal synthesis of thermally linked distillation columns
Model includes empirical relations for determining tray number, size, and cost of column

Includes disjunction definitions for intermediate and final product heat exchangers

Solvers are accesesd through GAMS

Spceies:
    A = Benzene
    B = Toluene
    C = EthylBenzene

Reference:
Caballero, J. A., & Grossmann, I. E. (2001). Generalized Disjunctive Programming Model
for the Optimal Synthesis of Thermally Linked Distillation Columns. Industrial & Engineering
Chemistry Research, 40(10), 2260-2274. https://doi.org/10.1021/ie000761a

"""

import sys
from math import pi
import random
import numpy as np
import pyomo.environ as pyo
from pyomo.util.infeasible import (
    log_infeasible_constraints,
    log_infeasible_bounds,
    find_infeasible_constraints,
)
from pyomo.gdp import Disjunction, Disjunct
from utils import (
    pprint_network,
    pprint_network_to_file,
    pprint_column,
    output_model,
    IntHeatExchanger,
    FinalHeatExchanger,
)


def build_model(stn):
    """
    Build a disjunctive model for the separation of a zeotropic mixture with thermally coupled distillation oclumns

    Args:
    stn : stn
        state-task network object that contains index sets for the separation of an N component mixture
    """

    N = stn.n  # number of components

    # MODEL DECLARATION
    # ================================================
    m = pyo.ConcreteModel(str(N) + " Component GDP Distillation Model")

    # PROBLEM DATA
    # ================================================

    # feed molar flow rate [kmol/hr]
    # 600 kmol/hr for the given hydrocarbon mixture is approximately 450 tons/yr
    F0 = 600

    # desired recovery of key components
    rec = 0.98

    P_abs = 1  # system pressure in [bara]
    Tf = 85  # system temp in [C]

    # feed molar fractions
    zf = {
        "A": 0.5,
        "B": 0.4,
        "C": 0.1,
    }

    # inlet molar component flow rates
    Fi0 = {key: value * F0 for key, value in zf.items()}

    # species relative volatilities
    relative_volatilty = {
        "A": 5.969,
        "B": 2.296,
        "C": 1,
    }

    # species liquid densities at 20 C in [kg/m^3]
    species_densities = {
        "A": 876,
        "B": 867,
        "C": 866,
    }

    # denisty of liquid mixture; assumed to be ideal and constant [kg/m^3]
    rho_L = sum(species_densities[key] * zf[key] for key in zf)

    # species molecular weights
    PM = {
        "A": 78,
        "B": 92,
        "C": 106,
    }

    # species vaporization enthalpy [KJ/mol]
    Hvap = {
        "A": 30.77,
        "B": 33.19,
        "C": 35.58,
    }

    # utility cost coefficients
    C_cw = 1.5e3 / 1e6  # cost of cooling utilities [$/kJ]
    C_h = 5.0e3 / 1e6  # cost of heating utilities [$/kJ]
    op_time = (
        24 * 360
    )  # assumed hours per year of column operating given some maintenance time

    # for use in calculating total annualized cost
    i = 0.08  # discount rate (interest rate)
    N = 30  # estimated lifetime of equipment in years

    # capital recovery factor: function of interest rate and lifetime of equipment
    CRF = (i * (1 + i) ** N) / ((1 + i) ** N - 1)

    # INDEX SETS
    # ================================================
    m.TASKS = pyo.Set(initialize=stn.TASKS,
                      doc="TASKS {t|t is a separation task}")

    m.STATES = pyo.Set(initialize=stn.STATES,
                       doc='mixture/states in the system')

    m.Feed = pyo.Set(initialize=stn.FEED,
                     doc='feed mixture to system')

    m.TS_s = pyo.Set(m.STATES, initialize=stn.TSs,
                     doc='TS_s = {tasks t that the state s is able to produce}')

    m.ST_s = pyo.Set(m.STATES, initialize=stn.STs,
                     doc='ST_s = {tasks t that are able to produce state s}')

    m.FS_F = pyo.Set(initialize=stn.FSf,
                     doc="FS_F = {Columns whose feed is the initial mixture}")

    m.PRE_i = pyo.Set(m.STATES, initialize=stn.PREi,
                      doc="PRE_i = {Tasks t that produce final product i through a rectifying section}")

    m.PST_i = pyo.Set(m.STATES, initialize=stn.PSTi,
                      doc="PST_i  = {Tasks t that produce final product i through a stripping section}")

    m.COMP = pyo.Set(initialize=stn.COMP,
                     doc="COM  ={i|i is a component in the mixture}")

    m.RECT_s = pyo.Set(m.STATES, initialize=stn.RECTs,
                       doc="RECT_s = {taks t that produces state s by a rectifying section}")

    # STRIP_s {taks t that produces state s by a stripping section}
    m.STRIP_s = pyo.Set(m.STATES, initialize=stn.STRIPs,
                        doc="STRIP_s {taks t that produces state s by a stripping section}")

    # Underwood roots
    # system has N = 3 components, thus will have at most 2 active underwood roots
    # roots are bounded by relative volatilities
    m.r = pyo.Set(initialize=stn.r, doc="Underwood roots")

    m.RUA = pyo.Set(m.TASKS, initialize=stn.RUA,
                    doc="active Underwood roots in column C")

    m.LK = pyo.Set(m.TASKS, initialize=stn.LK,
                   doc="light key (LK) component in a give separation task")

    m.HK = pyo.Set(m.TASKS, initialize=stn.HK,
                   doc="heavy key (HK) componenet in a given separation task")

    m.ISTATE = pyo.Set(initialize=stn.ISTATE,
                       doc="ISTATE = {m | m is an intermedaite state}")

    m.IREC_m = pyo.Set(m.STATES, initialize=stn.IRECs,
                       doc="IREC_m = {task t that produces intermediate state m from a rectifying section}")

    m.ISTRIP_m = pyo.Set(m.STATES, initialize=stn.ISTRIPs,
                         doc="ISTRIP_m = {task t that produces intermediate state m from a stripping section}")

    # CONTINUOUS POSITIVE VARIABLES
    # ================================================
    m.FT = pyo.Var(
        m.TASKS,
        doc="Total molar flow rate entering column k",
        within=pyo.NonNegativeReals,
        bounds=(0, F0),
    )

    m.F = pyo.Var(
        m.COMP,
        m.TASKS,
        doc="Component molar flow rate of species i entering column k",
        within=pyo.NonNegativeReals,
        bounds=(0, F0),
    )

    m.DT = pyo.Var(
        m.TASKS,
        doc="Total distillate molar flow rate of column k",
        within=pyo.NonNegativeReals,
        bounds=(0, F0),
    )

    m.D = pyo.Var(
        m.COMP,
        m.TASKS,
        doc="Component distillate molar flow rate of species i for column k",
        within=pyo.NonNegativeReals,
        bounds=(0, F0),
    )

    m.BT = pyo.Var(
        m.TASKS,
        doc="Total bottoms flow rate of column k",
        within=pyo.NonNegativeReals,
        bounds=(0, F0),
    )

    m.B = pyo.Var(
        m.COMP,
        m.TASKS,
        doc="Component bottoms flow rate of species i for column k",
        within=pyo.NonNegativeReals,
        bounds=(0, F0),
    )

    m.Vr = pyo.Var(
        m.TASKS,
        doc="Molar flow rate of vapor in the rectifying section of column k",
        within=pyo.NonNegativeReals,
        bounds=(0, F0),
        initialize=1,
    )

    m.Lr = pyo.Var(
        m.TASKS,
        doc="Molar flow rate of Liquid in the rectifying section of column k",
        within=pyo.NonNegativeReals,
        bounds=(0, F0),
    )

    m.Vs = pyo.Var(
        m.TASKS,
        doc="Molar flow rate of vapor in the stripping section of column k",
        within=pyo.NonNegativeReals,
        bounds=(0, F0),
    )

    m.Ls = pyo.Var(
        m.TASKS,
        doc="Molar flow rate of Liquid in the stripping section of column k",
        within=pyo.NonNegativeReals,
        bounds=(0, F0),
    )

    # Underwood root phi(t, r)
    m.rud = pyo.Var(
        m.TASKS,
        m.r,
        doc="possible active Underwood root (r) in task t",
        within=pyo.NonNegativeReals,
        bounds=(0, 500),
    )

    # Column costing and sizing variables
    m.column_cost = pyo.Var(
        m.TASKS,
        doc="Capital cost of column k",
        within=pyo.NonNegativeReals,
        bounds=(0, 10000000),
    )

    m.area_final_exchanger = pyo.Var(
        m.COMP,
        doc="heat exchange area of heat exchanger associated with final product [m^2]",
        within=pyo.NonNegativeReals,
        bounds=(0, 10000),
    )

    m.area_intermediate_exchanger = pyo.Var(
        m.COMP,
        doc="heat exchange area of heat exchanger associated with intermedaite product [m^2]",
        within=pyo.NonNegativeReals,
        bounds=(0, 10000),
    )

    m.final_condenser_cost = pyo.Var(
        m.COMP,
        doc="Capital cost of heat exchanger associated with final product i",
        within=pyo.NonNegativeReals,
        initialize=0,
        bounds=(0, 100000000),
    )

    m.final_reboiler_cost = pyo.Var(
        m.COMP,
        doc="Capital cost of heat exchanger associated with final product i",
        within=pyo.NonNegativeReals,
        initialize=0,
        bounds=(0, 100000000),
    )

    m.intermedaite_heat_exchanger_cost = pyo.Var(
        m.ISTATE,
        doc="Capital cost of heat exchanger associated with intermediate state s",
        within=pyo.NonNegativeReals,
        initialize=0,
        bounds=(0, 10000000),
    )

    m.Ntray = pyo.Var(
        m.TASKS,
        doc="Numer of trays in column (separation task) k",
        within=pyo.NonNegativeReals,
        bounds=(0, 200),
    )

    # columns height
    m.height = pyo.Var(
        m.TASKS,
        doc="height of column k in [m^2]",
        within=pyo.NonNegativeReals,
        bounds=(0, 100),
        initialize=1,
    )

    # column area
    m.Area = pyo.Var(
        m.TASKS,
        doc="Transveral area of column k",
        within=pyo.NonNegativeReals,
        bounds=(0, 1000),
        initialize=10,
    )

    # columns volume
    m.Vol = pyo.Var(
        m.TASKS,
        doc="volume of column k",
        within=pyo.NonNegativeReals,
        bounds=(0, 100000),
        initialize=10,
    )

    # reboiler heat duty
    m.Qreb = pyo.Var(
        m.TASKS,
        doc="reboiler heat duty for column k",
        within=pyo.NonNegativeReals,
        bounds=(0, 10000000),
        initialize=10,
    )

    # condenser heat duty
    m.Qcond = pyo.Var(
        m.TASKS,
        doc="condenser heat duty for column k",
        within=pyo.NonNegativeReals,
        bounds=(0, 10000000),
        initialize=10,
    )

    m.CAPEX = pyo.Var(
        doc="Total capital expense as sum of bare module purchase prices",
        within=pyo.NonNegativeReals,
        bounds=(0, 10000000),
        initialize=10,
    )

    m.OPEX = pyo.Var(
        doc="Total system operating expenses as sum of reboiler and condenser expenses",
        within=pyo.NonNegativeReals,
        bounds=(0, 10000000),
        initialize=10,
    )

    # PARAMETERS
    # ================================================

    m.F0 = pyo.Param(initialize=F0, doc="System inlet molar flow rate")

    m.F0_comp = pyo.Param(
        m.COMP, initialize=Fi0, doc="System inlet molar flow rate for each species i"
    )

    m.rec_comp = pyo.Param(
        m.COMP,
        initialize={"A": rec, "B": rec, "C": rec},
        doc="Specified recovery for each component",
    )

    m.alpha = pyo.Param(
        m.COMP, initialize=relative_volatilty, doc="species relative volatility"
    )

    m.PPM = pyo.Param(
        initialize=sum(zf[i] * PM[i] for i in m.COMP),
        doc="Molecular weight of feed stream",
    )

    m.rho_V = pyo.Param(
        m.TASKS,
        doc="Density of vapor in state s",
        within=pyo.NonNegativeReals,
        initialize=m.PPM * P_abs / 0.082 / (Tf + 273),
    )

    m.Hvap = pyo.Param(
        m.COMP, initialize=Hvap, doc="vaporization enthalpy for each species [KJ/mol]"
    )

    # GLOBAL CONSTRAINTS
    # ================================================

    @m.Constraint()
    def system_feed_total_mb(m):
        "Sum of feed inputs to columns that can take initial mixture must be same as system input"
        return sum(m.FT[t] for t in m.FS_F) == F0

    @m.Constraint(m.COMP)
    def system_feed_component_mb(m, i):
        "For each component, the fum of feed inputs to columns that can take initial mixture must be same as system input"
        return sum(m.F[(i, t)] for t in m.FS_F) == m.F0_comp[i]

    @m.Constraint(m.STATES)
    def total_mb_between_cols(m, s):
        """Constraint links the total molar outflows of distillate and bottoms of one column to the feed of another"""
        if s in m.Feed:
            return pyo.Constraint.Skip

        if s in m.TS_s:
            feed_flow = sum(m.FT[t] for t in m.TS_s[s])
            if s in m.RECT_s:
                dist_flow = sum(m.DT[t] for t in m.RECT_s[s])
            else:
                dist_flow = 0

            if s in m.STRIP_s:
                bot_flow = sum(m.BT[t] for t in m.STRIP_s[s])
            else:
                bot_flow = 0

            return feed_flow == dist_flow + bot_flow

        else:
            return pyo.Constraint.Skip

    @m.Constraint(m.STATES, m.COMP)
    def component_mb_between_cols(m, s, i):
        """Constraint links the component molar outflows of distillate and bottoms of one column to the feed of another"""
        if s in m.Feed:
            return pyo.Constraint.Skip

        if s in m.TS_s:
            feed_flow = sum(m.F[(i, t)] for t in m.TS_s[s])
            if s in m.RECT_s:
                dist_flow = sum(m.D[(i, t)] for t in m.RECT_s[s])
            else:
                dist_flow = 0

            if s in m.STRIP_s:
                bot_flow = sum(m.B[(i, t)] for t in m.STRIP_s[s])
            else:
                bot_flow = 0

            return feed_flow == dist_flow + bot_flow

        else:
            return pyo.Constraint.Skip

    @m.Constraint(m.STATES)
    def vapor_internal_mb(m, s):
        if s in m.Feed:
            return pyo.Constraint.Skip

        if s in m.TS_s:
            vapor_feed = sum(m.Vr[t] - m.Vs[t] for t in m.TS_s[s])
            if s in m.RECT_s:
                rect_vapor = sum(m.Vr[t] for t in m.RECT_s[s])
            else:
                rect_vapor = 0

            if s in m.STRIP_s:
                strip_vapor = sum(m.Vs[t] for t in m.STRIP_s[s])
            else:
                strip_vapor = 0

            return vapor_feed - rect_vapor + strip_vapor == 0

        else:
            return pyo.Constraint.Skip

    @m.Constraint(m.STATES)
    def liquid_internal_mb(m, s):
        if s in m.Feed:
            return pyo.Constraint.Skip

        if s in m.TS_s:
            liquid_feed = sum(m.Lr[t] - m.Ls[t] for t in m.TS_s[s])
            if s in m.RECT_s:
                rect_liquid = sum(m.Lr[t] for t in m.RECT_s[s])
            else:
                rect_liquid = 0

            if s in m.STRIP_s:
                strip_liquid = sum(m.Ls[t] for t in m.STRIP_s[s])
            else:
                strip_liquid = 0

            return liquid_feed - rect_liquid + strip_liquid == 0

        else:
            return pyo.Constraint.Skip

    @m.Constraint(m.COMP)
    def final_product_mb(m, i):
        """constraint linking final product flows based on a specified recovery to the distillate
        and bottoms flows of columsn that can produce that final state"""

        if i in m.PRE_i:
            final_dist = sum(m.D[(i, k)] for k in m.PRE_i[i])
        else:
            final_dist = 0

        if i in m.PST_i:
            final_bot = sum(m.B[(i, k)] for k in m.PST_i[i])
        else:
            final_bot = 0

        return final_dist + final_bot >= m.rec_comp[i] * m.F0_comp[i]

    # DISJUNCTS FOR COLUMNS (SEPARATION TASKS)
    # ================================================
    m.column = Disjunct(m.TASKS, doc="Disjunct for column existence")
    m.no_column = Disjunct(m.TASKS, doc="Disjunct for column absence")

    # column disjunction
    @m.Disjunction(m.TASKS, doc="Column exists or does not")
    def column_no_column(m, k):
        return [m.column[k], m.no_column[k]]

    # Functions for defining mass balance and Underwood relation constraints
    # ================================================

    def _build_mass_balance_column(m, t, column):
        """
        Function to build mass balance relation for active column disjuncts

        Args:
            m(pyomo.ConcreteModel): Pyomo model object contains relevant variables, parameters
                and expressions for distillation network
            t (str): index from the set TASKS representing the column
            column (pyomo.gdp.disjunct): Pyomo Disjunct representing an active column

        Constraints:
            -column_total_mb: mass balance on the total molar flows of the active column
            -column_component_mb: mass balance on the component molar flows of the active column
            -column_rectifying_mb: mass balance on the total flows in the rectifying section of the active column
            -column_strippiong_mb: mass balance on the total flows in the stripping section of the active column
            -feed_comp_mb: sum of component flows equals total flow for column feed
            -distillate_comp_mb: sum of component flows equals total flow for column distillate
            -bottoms_comp_mb: sum of component flows equals total flow for column bottoms

        Returns:
            None: The function directly updates the model object, adding constraints to it.
        """

        @column.Constraint()
        def column_total_mb(_):
            return m.FT[t] == m.DT[t] + m.BT[t]

        @column.Constraint(m.COMP)
        def column_component_mb(_, i):
            """Component mass balance for feed and distillate in column disjunct k"""
            return m.F[(i, t)] == m.D[(i, t)] + m.B[(i, t)]

        @column.Constraint()
        def column_rectifying_mb(_):
            """Total mass balance on rectifying section of column"""
            return m.DT[t] + m.Lr[t] == m.Vr[t]

        @column.Constraint()
        def column_stripping_mb(_):
            """Total mass balance on stripping section of column"""
            return m.BT[t] + m.Vs[t] == m.Ls[t]

        @column.Constraint()
        def feed_comp_mb(_):
            """Sum of component flows into column equals total flow"""
            return m.FT[t] == sum(m.F[(i, t)] for i in m.COMP)

        @column.Constraint()
        def distillate_comp_mb(_):
            """Sum of component flows in distillate equals total distillate flow"""
            return m.DT[t] == sum(m.D[(i, t)] for i in m.COMP)

        @column.Constraint()
        def bottoms_comp_mb(_):
            """Sum of component flows in bottoms equals total bottoms flow"""
            return m.BT[t] == sum(m.B[(i, t)] for i in m.COMP)

    def _build_mass_balance_no_column(m, t, nocolumn):
        """
        Function to build mass balance relation for inactive column disjuncts

        Args:
            m(pyomo.ConcreteModel): Pyomo model object contains relevant variables, parameters
                and expressions for distillation network
            t (str): index from the set TASKS representing the column
            no_column (pyomo.gdp.disjunct): Pyomo Disjunct representing an inactive column

        Constraints:
            Constraints set all total and component molar flows for inactive column
            disjunct to be zero

        Returns:
        None: The function directly updates the model object, adding constraints to it.
        """

        @nocolumn.Constraint(m.COMP)
        def inactive_feed_component_flow(_, i):
            return m.F[(i, t)] == 0

        @nocolumn.Constraint(m.COMP)
        def inactive_distillate_component_flow(_, i):
            return m.D[(i, t)] == 0

        @nocolumn.Constraint(m.COMP)
        def inactive_bottoms_component_flow(_, i):
            return m.B[(i, t)] == 0

        @nocolumn.Constraint()
        def inactive_vapor_rectifying_flow(_):
            return m.Vr[t] == 0

        @nocolumn.Constraint()
        def inactive_liquid_rectifying_flow(_):
            return m.Lr[t] == 0

        @nocolumn.Constraint()
        def inactive_liquid_stripping_flow(_):
            return m.Ls[t] == 0

        @nocolumn.Constraint()
        def inactive_feed_total_flow(_):
            return m.FT[t] == 0

        @nocolumn.Constraint()
        def inactive_distillate_total_flow(_):
            return m.DT[t] == 0

        @nocolumn.Constraint()
        def inactive_bottoms_total_flow(_):
            return m.BT[t] == 0

    def _build_underwood_eqns(m, t, column):
        """
        Function to build underwood equations for a given column

        Args:
            m(pyomo.ConcreteModel): Pyomo model object contains relevant variables, parameters
                and expressions for distillation network
            t (str): index from the set TASKS representing the column
            column (pyomo.gdp.disjunct): Pyomo Disjunct representing an active column

        Constraints:
            -underwood1,2,3: Underwood equations act as shortcut models for operation of column in terms of
            relative volatilities of species

        Returns:
        None: The function directly updates the model object, adding constraints to it.
        """

        roots = list(m.RUA[t])

        @column.Constraint(roots)
        def underwood1(_, r):
            return sum((m.alpha[i] * m.F[(i, t)]) for i in m.COMP) == (
                m.Vr[t] - m.Vs[t]
            ) * sum(m.alpha[i] - m.rud[(t, r)] for i in m.COMP)

        @column.Constraint(roots)
        def underwood2(_, r):
            return sum((m.alpha[i] * m.D[(i, t)]) for i in m.COMP) <= m.Vr[t] * sum(
                m.alpha[i] - m.rud[(t, r)] for i in m.COMP
            )

        @column.Constraint(roots)
        def underwood3(_, r):
            return -sum((m.alpha[i] * m.B[(i, t)]) for i in m.COMP) <= m.Vs[t] * sum(
                m.alpha[i] - m.rud[(t, r)] for i in m.COMP
            )

    # Functions for defining tray number, column size, and cost constraints
    # ================================================
    def _build_trays(m, t, column):
        """
        Builds constraints for the number of trays associated with each separation task t using Fenske equation.
        Fenske equation gives a minimum number of trays. Actual trays is give as twice the minimum number

        Args:
            m(pyomo.ConcreteModel): Pyomo model object contains relevant variables, parameters
                and expressions for distillation network
            t (str): index from the set TASKS representing the column
            column (pyomo.gdp.disjunct): Pyomo Disjunct representing an active column

        Constraints:
            -Number_trays: calculates actual number of trays associated with each separation task as a multiple
            of the minimum tray number from the Fenske equation

        Return:
            None. Function adds constraints to the model but does not return a value
        """

        @column.Constraint()
        def Number_trays(_):
            """ "Minimum tray number from empirical correlation. Actual number of trays as twice the minimum number"""
            Ntray_min = pyo.log10(rec**2 / (1 - rec) ** 2) / pyo.log10(
                sum(m.alpha[i] for i in m.LK[t]) / sum(m.alpha[i] for i in m.HK[t])
            )
            return m.Ntray[t] == 2 * Ntray_min

    def _build_column_height(m, t, column):
        @column.Constraint()
        def column_height(_):
            """Empirical relation for the height of the column from Ulrich et al Ch 4. Column height in [m^2]"""
            D = 2 * pyo.sqrt(m.Area[t] / pi)
            H_per_tray = 0.5 * D**0.3
            return m.height[t] == m.Ntray[t] * H_per_tray

    def _build_column_area(m, t, column):
        """
        Builds constraints for the column area for each column disjunct

        Args:
            m(pyomo.ConcreteModel): Pyomo model object contains relevant variables, parameters
                and expressions for distillation network
            t (str): index from the set TASKS representing the column
            column (pyomo.gdp.disjunct): Pyomo Disjunct representing an active column

        Constraints:
            -column_area: calculation of column area based on empirical correlation

        Return:
            None. Function adds constraints to the model but does not return a value
        """

        def column_area_correlation(V_dot, rho_L, rho_g, PM, sigma=0.02, phi=0.1, dh=8):
            """Function calcaultes the area of a sieve tray column for a mixture at a fixed pressure.
            Assumes that trays have large spacing and assumed value for relative weir length

            Args:
            -V_dot (float): Vapor molar flow rate [kmol/hr]
            -PM (float): Molecular weight of mixture [kg/kmol]
            -rho_L (float): Density of liquid [kg/m^3]
            -rho_g (float): Density of gas [kg/m^3]
            -sigma (float): Surface tension of mixture [N/m]
            -phi: relative free area; default of 0.1
            -dh: tray hole diameter in mm; default value of 8 mm

            Returns:
            -Ac (float): Column cross sectional area [m^2]

            Calcuation procedure from Chapter 9.2 of:
            Stichlmair, J. G., Klein, H., & Rehfeldt, S. (2021). Distillation Principles and Practice.
            Newark American Institute Of Chemical Engineers Ann Arbor, Michigan Proquest.
            """

            g = 9.81  # gravitational constant [m/sec^2]
            dh = dh / 1000  # unit conversion of sieve hole diamter from mm to meters
            weir_length = 0.7  # use of a relative weir length of 0.7
            V_dot = V_dot * (1 / 3600)  # conversion of units to [kmol/sec]

            # maxium gas load for large tray spacing
            F_max = 2.5 * (phi**2 * sigma * (rho_L - rho_g) * g) ** (1 / 4)

            """can use 2 criterion for minimum gas load:
            F_min_1: non-uniform gas flow
            F_min_2: criterion for weeping"""

            F_min_1 = phi * np.sqrt(2 * (sigma / dh))
            F_min_2 = phi * np.sqrt(
                0.37 * dh * g * ((rho_L - rho_g) ** 1.25) / rho_g**0.25
            )

            # take the max of the 2; using logical comparison to avoid nondifferentiable max() operator
            if F_min_1 > F_min_2:
                F_min = F_min_1
            elif F_min_2 >= F_min_2:
                F_min = F_min_2

            # take the geometric mean of min and max gas load to get gas load specification for system
            F = (F_min * F_max) ** (1 / 2)

            u_g = F / np.sqrt(rho_g)  # calcuating sueprficial gas velocity

            V_g = V_dot * (
                PM / rho_g
            )  # calculation of volumetric gas flow rate [m^3/sec]
            Aac = V_g / u_g  # calculation of active area of a tray [m^2]

            temp = 1 - (2 / np.pi) * (
                np.arcsin(weir_length) - np.sqrt(weir_length**2 - weir_length**4)
            )
            Ac = Aac / temp  # column area [m^2]

            return Ac  # column area in [m^2]

        @column.Constraint()
        def column_area(_):
            """Calculation of column area based on empirical correlation"""
            return m.Area[t] == column_area_correlation(
                m.Vr[t], rho_L, m.rho_V[t], m.PPM
            )

    def _build_column_volume(m, t, column):
        """
        Builds constraints for the volume of each column

        Args:
            m(pyomo.ConcreteModel): Pyomo model object contains relevant variables, parameters
                and expressions for distillation network
            t (str): index from the set TASKS representing the column
            column (pyomo.gdp.disjunct): Pyomo Disjunct representing an active column

        Constraints:
            -column_volume: calculation of column volume based on empirical correlation

        Return:
            None. Function adds constraints to the model but does not return a value
        """

        @column.Constraint()
        def columnn_volume(_):
            """calculation of column volume based on empirical correlation"""
            return m.Vol[t] == m.Area[t] * m.height[t]

    def _build_no_column_size(m, t, nocolumn):
        """Function builds constraints that set physical parameters of column (tray number, heigh,
        area, and volume) to zero for an inactive column"""

        @nocolumn.Constraint()
        def inactive_trays(_):
            return m.Ntray[t] == 0

        @nocolumn.Constraint()
        def inactive_height(_):
            return m.height[t] == 0

        @nocolumn.Constraint()
        def inactive_area(_):
            return m.Area[t] == 0

        @nocolumn.Constraint()
        def inactive_volume(_):
            return m.Vol[t] == 0

    def _build_column_cost(m, k, column, no_column):
        """
        Builds constraints for the cost relation for each column

        Args:
            m(pyomo.ConcreteModel): Pyomo model object contains relevant variables, parameters
                and expressions for distillation network
            t (str): index from the set TASKS representing the column
            column (pyomo.gdp.disjunct): Pyomo Disjunct representing an active column
            no_column (pyomo.gdp.disjunct): Pyomo Disjunct representing an inactive column

        Constraints:
            -column_cost: empirical correlation based on calculcations in Turton et al.
                column cost as sum of cost of trays and vertical process vessel
            -no_column_cost: capital cost of column set to zero for inactive disjunct

        Return:
            None. Function adds constraints to the model but does not return a value
        """

        @column.Constraint()
        def column_cost(_):
            # values used to calculate price changes due to inflation
            CEPCI_2004 = 444.2
            CEPCI_2020 = 596.2
            inflation_ratio = CEPCI_2020 / CEPCI_2004

            Cpisos = 571.1 + 406.8 * m.Area[k] + 38 * m.Area[k] ** 2
            CostPisos = Cpisos * m.Ntray[k]

            CP = 603.8 * m.Vol[k] + 5307
            CostColumn = CP * (2.50 + 1.72)

            return m.column_cost[k] == (CostColumn + CostPisos) * inflation_ratio

        @no_column.Constraint()
        def no_column_cost(_):
            return m.column_cost[k] == 0

    # building out constraints for columns with functions
    # =================================================================

    for t in m.TASKS:
        _build_mass_balance_column(m, t, m.column[t])
        _build_mass_balance_no_column(m, t, m.no_column[t])
        _build_underwood_eqns(m, t, m.column[t])
        _build_trays(m, t, m.column[t])
        _build_column_height(m, t, m.column[t])
        _build_column_area(m, t, m.column[t])
        _build_column_volume(m, t, m.column[t])
        _build_no_column_size(m, t, m.no_column[t])
        _build_column_cost(m, t, m.column[t], m.no_column[t])

    # DISJUNCTS FOR FINAL PRODUCT HEAT EXCHANGERS
    # ================================================
    m.final_heat_exchanger = Disjunct(
        m.COMP,
        doc="Disjunct for the existence of a heat exchanger associated with a final product",
    )
    m.no_final_heat_exchanger = Disjunct(
        m.COMP,
        doc="Disjunct for the non-existence of a heat exchanger associated with a final product",
    )

    # final product heat exchanger disjunction are indexed by pure components i in COMPS [e.g. A, B, C]
    @m.Disjunction(m.COMP, doc="Final product heat exchanger exists or does not")
    def final_heat_exchange_no_final_heat_exchange(m, i):
        return [m.final_heat_exchanger[i], m.no_final_heat_exchanger[i]]

    def _build_final_product_heat_exchanger(m, i, heat_exchanger, no_heat_exchanger):
        """Function to build constraints for disjuncts of final product heat exchangers

        Args:
            m(pyomo.ConcreteModel): Pyomo model object contains relevant variables, parameters
                and expressions for distillation network
            i (str): index from the set COMP a final product (A, B, C, etc.)
            heat_exchanger (pyomo.gdp.disjunct): Pyomo Disjunct representing an active heat exchanger for a final product
            no_heat_exchanger (pyomo.gdp.disjunct): Pyomo Disjunct representing an inactive heat exchanger for a final product

        Constraints:
            -condenser_heat_duty: Qcond is a function of species vaporization enthalpy and vapor flow rates
            -reboiler_heat_duty: Qreb is a function of species vaporization enthalpy and vapor flow rates
            inactive_final_condenser: Qcond = 0 if the final state is not produced by a rectifying section
            inactive_final_reboiler: Qreb = 0 if the final state is not produced by a stripping section

        Return:
            None. Function adds constraints to the model but does not return a value
        """

        # Constraints for heat exchangers associated with final states produced by a rectifying section (reobilers)
        if i in m.PRE_i:
            rect_tasks = m.PRE_i[i]

            @heat_exchanger.Constraint(rect_tasks)
            def condenser_heat_duty(_, t):
                """Task reboiler heat duty based on enthalpy of vaporization and stripping section vapor flow rate"""
                return m.Qcond[t] * m.DT[t] == sum(
                    m.Hvap[j] * m.D[(j, t)] * m.Vr[t] for j in m.COMP
                )

            @no_heat_exchanger.Constraint(rect_tasks)
            def inactive_final_condenser(_, t):
                """final product heat exchange for a stream from a rectifying section is not selected"""
                return m.Qcond[t] == 0

        # Constraints for heat exchangers associated with final states produced by a stripping section (condensers)
        if i in m.PST_i:
            strip_tasks = m.PST_i[i]

            @heat_exchanger.Constraint(strip_tasks)
            def reboiler_heat_duty(_, t):
                """Task condenser heat duty based on enthalpy of vaporization and rectifying section vapor flow rate"""
                return m.Qreb[t] * m.BT[t] == sum(
                    m.Hvap[j] * m.B[(j, t)] * m.Vr[t] for j in m.COMP
                )

            @no_heat_exchanger.Constraint(strip_tasks)
            def inactive_final_reboiler(_, t):
                """final product heat exchange for a stream from a stripping section is not selected"""
                return m.Qreb[t] == 0

    def _final_heat_exchanger_cost(m, i, heat_exchanger):
        """Function to build constraints for capital cost of active final product heat exchanger

        Args:
            m(pyomo.ConcreteModel): Pyomo model object contains relevant variables, parameters
                and expressions for distillation network
            i (str): index from the set COMP a final product (A, B, C, etc.)
            heat_exchanger (pyomo.gdp.disjunct): Pyomo Disjunct representing an active heat exchanger for a final product

        Constraints:
            -final_excahnger_area:
            -final_excahnger_cost:

        Return:
            None. Function adds constraints to the model but does not return a value
        """
        # Data for cost correlation
        # Chemical Engineering Plant Cost Index values for selected years
        CEPCI_2004 = 400
        CEPCI_2020 = 596.2
        inflation_ratio = CEPCI_2020 / CEPCI_2004

        n = 0.6  # cost relation exponent

        # for heat exchangers with 100 [m^2] of exchange area; Carbon steel construction
        # source: Ulrich et al 5.36
        Reboiler_base_price_2004 = 20000
        Shell_tube_base_price_2004 = 15000

        C_reb_2020 = Reboiler_base_price_2004 * inflation_ratio
        C_shell_tube_2020 = Shell_tube_base_price_2004 * inflation_ratio

        K_reb = C_reb_2020 / (100) ** n
        K_shell_tube = C_shell_tube_2020 / (100) ** n

        # U for hot side of condensing steam and cold side fluid of low viscosity liquid hydrocarbons
        Ureb = 600  # [J / m^2 sec K]
        # U for hot side of hydrocarbon gases and cold side fluid of liquid water
        Ucond = 150  # [J / m^2 sec K]

        # delta log mean temp difference from condenser and reboiler
        delta_LM_T_cond = 50
        delta_LM_T_reb = 270

        # if the final state is produced by a rectifying section, will have assocaited condenser
        if i in m.PRE_i:

            @heat_exchanger.Constraint()
            def condenser_area(_):
                Qcon_inter = (
                    sum(m.Qcond[t] for t in m.PRE_i[i]) * 1000
                )  # multiply by 1000 for unit conversion to [J/se0c]
                return m.area_final_exchanger[i] == Qcon_inter / (
                    Ucond * delta_LM_T_cond
                )

            @heat_exchanger.Constraint()
            def condenser_cost(_):
                return (
                    m.final_condenser_cost[i]
                    == K_shell_tube * m.area_final_exchanger[i] ** n
                )

        # if the final state is produced by a stripping section, will have assocaited reboiler
        if i in m.PST_i:

            @heat_exchanger.Constraint()
            def reboiler_area(_):
                Qreb_inter = (
                    sum(m.Qreb[t] for t in m.PST_i[i]) * 1000
                )  # multiply by 1000 for unit conversion to [J/se0c]
                return m.area_final_exchanger[i] == Qreb_inter / (Ureb * delta_LM_T_reb)

            @heat_exchanger.Constraint()
            def reboiler_cost(_):
                return (
                    m.final_reboiler_cost[i] == K_reb * m.area_final_exchanger[i] ** n
                )

    for i in m.COMP:
        _build_final_product_heat_exchanger(
            m, i, m.final_heat_exchanger[i], m.no_final_heat_exchanger[i]
        )
        _final_heat_exchanger_cost(m, i, m.final_heat_exchanger[i])

    # DISJUNCTS FOR INTERMEDIATE PRODUCT HEAT EXCHANGERS
    # ================================================
    m.int_heat_exchanger = Disjunct(
        m.ISTATE,
        doc="Disjunct for the existence of a heat exchanger associated with a intermedaite product",
    )
    m.no_int_heat_exchanger = Disjunct(
        m.ISTATE,
        doc="Disjunct for the non-existence of a heat exchanger associated with a intermedaite product",
    )

    # intermediate state heat exchanger disjunctions are indexed by ISTATE [e.g. AB, BC]
    @m.Disjunction(
        m.ISTATE, doc="Intermedaite product heat exchanger exists or does not"
    )
    def int_heat_exchange_no_final_heat_exchange(m, i):
        return [m.int_heat_exchanger[i], m.no_int_heat_exchanger[i]]

    def _build_intermediate_product_heat_exchanger(m, s, heat_exchanger):
        """Function to build constraints for disjuncts of active intermediate product heat exchangers

        Args:
            m(pyomo.ConcreteModel): Pyomo model object contains relevant variables, parameters
                and expressions for distillation network
            i (str): index from the set COMP a final product (A, B, C, etc.)
            heat_exchanger (pyomo.gdp.disjunct): Pyomo Disjunct representing an active heat exchanger for a final product

        Constraints:
            -int_condenser_heat_duty: Qcond is a function of species vaporization enthalpy and vapor flow rates
            -int_reboiler_heat_duty: Qreb is a function of species vaporization enthalpy and vapor flow rates

        Return:
            None. Function adds constraints to the model but does not return a value
        """

        # if intermediate state is produced by a rectifying section and it has a heat exchanger, it will be a condenser
        if s in m.IREC_m:
            rect_tasks = m.IREC_m[s]

            @heat_exchanger.Constraint(rect_tasks)
            def int_condenser_heat_duty(_, t):
                """Task reboiler heat duty based on enthalpy of vaporization and stripping section vapor flow rate"""
                return m.Qcond[t] * m.DT[t] == sum(
                    m.Hvap[i] * m.D[(i, t)] * m.Vr[t] for i in m.COMP
                )

        # if intermediate state is produced by a stripping section and it has a heat exchanger, it will be a reboiler
        if s in m.ISTRIP_m:
            strip_tasks = m.ISTRIP_m[s]

            @heat_exchanger.Constraint(strip_tasks)
            def int_reboiler_heat_duty(_, t):
                """Task condenser heat duty based on enthalpy of vaporization and rectifying section vapor flow rate"""
                return m.Qreb[t] * m.BT[t] == sum(
                    m.Hvap[i] * m.B[(i, t)] * m.Vs[t] for i in m.COMP
                )

    def _build_intermediate_heat_exchanger_cost(m, s, heat_exchanger):
        """Function to build constrait to define the capital cost of heat exchanger

        Args:
            m(pyomo.ConcreteModel): Pyomo model object contains relevant variables, parameters
                and expressions for distillation network
            s (str): index from the set COMP a final product (A, B, C, etc.)
            heat_exchanger (pyomo.gdp.disjunct): Pyomo Disjunct representing an active heat exchanger for a final product

        Constraints:
            -int_condenser_heat_duty: Qcond is a function of species vaporization enthalpy and vapor flow rates
            -int_reboiler_heat_duty: Qreb is a function of species vaporization enthalpy and vapor flow rates

        Return:
            None. Function adds constraints to the model but does not return a value
        """

    def _build_inactive_intermediate_product_heat_exchanger(m, s, no_heat_exchanger):
        """Function to build constraints for disjuncts of inactive intermediate product heat exchangers

        Args:
            m(pyomo.ConcreteModel): Pyomo model object contains relevant variables, parameters
                and expressions for distillation network
            i (str): index from the set COMP a final product (A, B, C, etc.)
            heat_exchanger (pyomo.gdp.disjunct): Pyomo Disjunct representing an active heat exchanger for a final product

        Constraints:
            -inactive_intermediate_condenser: Qcond = 0
            -inactive_intermediate_reboiler: Qreb = 0
            -heat_exchanger_mb_vapor: mass balance relation for flow through of vapor streams for inactive heat exchanger
            -heat_exchanger_mb_liquid: mass balance relation for flow through of liquid streams for inactive heat exchanger
            -no_heat_exchanger_cost: Cost[s] = 0

        Return:
            None. Function adds constraints to the model but does not return a value
        """

        if s in m.IREC_m:
            rect_tasks = m.IREC_m[s]

            @no_heat_exchanger.Constraint(rect_tasks)
            def inactive_intermediate_condenser(_, t):
                """final product heat exchange for a stream from a rectifying section is not selected"""
                return m.Qcond[t] == 0

        if s in m.ISTRIP_m:
            strip_tasks = m.ISTRIP_m[s]

            @no_heat_exchanger.Constraint(strip_tasks)
            def inactive_intermediate_reboiler(_, t):
                """final product heat exchange for a stream from a stripping section is not selected"""
                return m.Qreb[t] == 0

        if s in m.IREC_m:
            vapor_rec_flow = sum(m.Vr[t] for t in m.IREC_m[s])
            liquid_rec_flow = sum(m.Lr[t] for t in m.IREC_m[s])
        else:
            vapor_rec_flow = 0
            liquid_rec_flow = 0

        if s in m.ISTRIP_m:
            vapor_strip_flow = sum(m.Vs[t] for t in m.ISTRIP_m[s])
            liquid_strip_flow = sum(m.Ls[t] for t in m.ISTRIP_m[s])
        else:
            vapor_strip_flow = 0
            liquid_strip_flow = 0

        @no_heat_exchanger.Constraint()
        def heat_exchanger_mb_vapor(_):
            return (
                sum(m.Vr[t] - m.Vs[t] for t in m.TS_s[s]) + vapor_strip_flow
                == vapor_rec_flow
            )

        @no_heat_exchanger.Constraint()
        def heat_exchanger_mb_liquid(_):
            return (
                sum(m.Lr[t] - m.Ls[t] for t in m.TS_s[s]) + liquid_strip_flow
                == liquid_rec_flow
            )

        @no_heat_exchanger.Constraint()
        def no_heat_exchanger_cost(_):
            return m.intermedaite_heat_exchanger_cost[s] == 0

    # function calls for intermediate product heat exchangers
    for s in m.ISTATE:
        _build_intermediate_product_heat_exchanger(m, s, m.int_heat_exchanger[s])
        _build_inactive_intermediate_product_heat_exchanger(
            m, s, m.no_int_heat_exchanger[s]
        )

    # LOGICAL CONSTRAINTS
    # ================================================

    @m.Constraint(
        m.STATES,
        doc="""A given state s can give rise to at most one task: cannot split a product
                stream and send to 2 different columns""",
    )
    def logic1(m, s):
        if s in m.ST_s:
            return sum(m.column[t].binary_indicator_var for t in m.ST_s[s]) <= 1
        else:
            return pyo.Constraint.Skip

    """Logic 2 and Logic 3: A given state can be produced by at most 2 tasks; one must
    be from a rectifying section and one must be from a stripping section"""

    @m.Constraint(m.STATES, doc="State generated by rectifying section")
    def logic2(m, s):
        if s in m.RECT_s:
            tasks = list(m.RECT_s[s])
            return sum(m.column[t].binary_indicator_var for t in tasks) <= 1
        else:
            return pyo.Constraint.Skip

    @m.Constraint(m.STATES, doc="State generated by stripping section")
    def logic3(m, s):
        if s in m.STRIP_s:
            tasks = list(m.STRIP_s[s])
            return sum(m.column[t].binary_indicator_var for t in tasks) <= 1
        else:
            return pyo.Constraint.Skip

    @m.Constraint(m.COMP, doc="All products must be produced by at least one task")
    def logic4(m, i):
        if i in m.PRE_i:
            task1 = m.PRE_i[i]
        else:
            task1 = []
        if i in m.PST_i:
            task2 = m.PST_i[i]
        else:
            task2 = []
        return (
            sum(m.column[t].binary_indicator_var for t in task1)
            + sum(m.column[t].binary_indicator_var for t in task2)
            >= 1
        )

    # Pure product i can only be produced by at most one rectifying section and one stripping section
    @m.Constraint(
        m.COMP,
        doc="Pure product i can only be produced by at most 1 rectifying section",
    )
    def logic5(m, i):
        if i in m.PRE_i:
            return sum(m.column[t].binary_indicator_var for t in m.PRE_i[i]) <= 1
        else:
            return pyo.Constraint.Skip

    @m.Constraint(
        m.COMP, doc="Pure product i can only be produced by at most 1 stripping section"
    )
    def logic6(m, i):
        if i in m.PST_i:
            return sum(m.column[t].binary_indicator_var for t in m.PST_i[i]) <= 1
        else:
            return pyo.Constraint.Skip

    """Logic 7 and Logic 8: If a final product is produced by exactly one contribution, the heat exchanger associated with
    this product must be selected."""

    @m.LogicalConstraint(m.COMP)
    def logic7(m, i):
        if i in m.PRE_i:
            bool_vars = [m.column[t].indicator_var for t in m.PRE_i[i]]
            return pyo.implies(
                pyo.exactly(1, bool_vars), m.final_heat_exchanger[i].indicator_var
            )
        else:
            return pyo.Constraint.Skip

    @m.LogicalConstraint(m.COMP)
    def logic8(m, i):
        if i in m.PST_i:
            bool_vars = [m.column[t].indicator_var for t in m.PST_i[i]]
            return pyo.implies(
                pyo.exactly(1, bool_vars), m.final_heat_exchanger[i].indicator_var
            )
        else:
            return pyo.Constraint.Skip

    """If a given final state is produced by 2 tasks, then there is no heat exchnager associated with that state"""

    @m.LogicalConstraint(m.COMP)
    def logic9(m, i):
        if i in m.PRE_i:
            temp_var1 = [m.column[t].indicator_var for t in m.PRE_i[i]]
        else:
            temp_var1 = []
        if i in m.PST_i:
            temp_var2 = [m.column[t].indicator_var for t in m.PST_i[i]]
        else:
            temp_var2 = []

        return pyo.implies(
            pyo.land(temp_var1, temp_var2),
            pyo.lnot(m.final_heat_exchanger[i].indicator_var),
        )

    """Intermediate heat exchanger logic: cannot have a heat exchanger for an intermediate state if the state is not produced by a task"""

    @m.Constraint(m.ISTATE)
    def logic10(m, s):
        return (1 - m.int_heat_exchanger[s].binary_indicator_var + sum(m.column[t].binary_indicator_var for t in m.ST_s[s]) >= 1)

    @m.LogicalConstraint(m.STATES, m.TASKS,
                         doc="""Connectivity relations e.g. Existence of task AB/C implies A/B""")
    def logic11(m, s, t):
        if s in m.ST_s and s in m.TS_s:
            tasks = list(m.ST_s[s])
            if t in tasks:
                some_list = [m.column[k].indicator_var for k in m.TS_s[s]]
                return pyo.implies(m.column[t].indicator_var, pyo.lor(some_list))
            else:
                return pyo.Constraint.Skip
        else:
            return pyo.Constraint.Skip

    @m.LogicalConstraint(m.STATES, m.TASKS,
                         doc="""Connectivity relations e.g. A/B implies AB/C""")
    def logic12(m, s, t):
        if s in m.TS_s and s in m.ST_s:
            tasks = list(m.TS_s[s])
            if t in tasks:
                some_list = [m.column[k].indicator_var for k in m.ST_s[s]]
                return pyo.implies(m.column[t].indicator_var, pyo.lor(some_list))
            else:
                return pyo.Constraint.Skip
        else:
            return pyo.Constraint.Skip

    # CONSTRAINTS FOR DEFINING OBJECTIVE FUNCTION
    # ================================================
    @m.Constraint()
    def capex_def(m):
        return m.CAPEX == sum(m.column_cost[k] for k in m.TASKS)

    @m.Constraint()
    def opex_def(m):
        return m.OPEX == C_cw * op_time * sum(m.Qreb[t] for t in m.TASKS) + C_h * op_time * sum(m.Qcond[t] for t in m.TASKS)

    # OBJECTIVE
    # ================================================
    # multiply sum of bare module capital expenses by CRF to get annualized cost
    m.obj = pyo.Objective(expr=CRF * m.CAPEX + m.OPEX, sense=pyo.minimize)

    return m