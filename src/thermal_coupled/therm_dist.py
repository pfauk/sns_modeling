"""
GDP Model for the optimal synthesis of thermally linked distillation columns
Model includes empirical relations for determining tray number, size, and cost of column

Superstructure generation for separation network done in separate script

Includes disjunction definitions for intermediate and final product heat exchangers

Current problem is reformulated to be an MIQCP to solve with Gurobi

References:
Caballero, J. A., & Grossmann, I. E. (2001). Generalized Disjunctive Programming Model
for the Optimal Synthesis of Thermally Linked Distillation Columns. Industrial & Engineering
Chemistry Research, 40(10), 2260-2274. https://doi.org/10.1021/ie000761a

Caballero, J. A., & Grossmann, I. E. (2004). Design of distillation sequences: From conventional to
fully thermally coupled distillation systems.Computers & Chemical Engineering,
28(11), 2307–2329. https://doi.org/10.1016/j.compchemeng.2004.04.010

"""

from math import pi
import pyomo.environ as pyo
from pyomo.gdp import Disjunct

def build_model(stn, data)->pyo.ConcreteModel:
    """
    Build a disjunctive model for the separation of a zeotropic mixture with thermally coupled distillation columns

    Args:
    stn : stn
        state-task network object that contains index sets for the separation of an N component mixture
    data : data
        data object that contains relevant species properties and systems specifications
    """

    N = stn.n  # number of components

    if N != data.n:
        raise ValueError('Number of components in STN object must be the same as number of components in data sheet')

    # MODEL DECLARATION
    # ================================================
    m = pyo.ConcreteModel(str(N) + " Component GDP Distillation Model")

    # PROBLEM DATA
    # ================================================

    # total system feed molar flow rate [kmol/hr]
    F0 = data.F0

    # desired recovery of each components
    comp_rec = data.rec

    P_abs = data.P_abs  # system pressure in [bara]
    Tf = data.Tf  # system temp in [C]

    # feed molar fractions
    zf = data.zf

    # inlet molar component flow rates
    Fi0 = {key: value * F0 for key, value in zf.items()}

    # species relative volatilities
    relative_volatility = data.relative_volatility

    # species liquid densities at 20 C in [kg/m^3]
    species_densities = data.species_densities

    # density of liquid mixture; assumed to be ideal and constant [kg/m^3]
    rho_L = sum(species_densities[key] * zf[key] for key in zf)

    # species molecular weights
    PM = data.PM

    # species vaporization enthalpy [KJ/mol]
    Hvap = data.Hvap

    # utility cost coefficients
    C_cw = data.C_cw  # cost of cooling utilities [$/kJ]
    C_h = data.C_h  # cost of heating utilities [$/kJ]
    op_time = 24 * 360  # assumed hours per year of column operating given some maintenance time

    # for use in calculating total annualized cost
    i = 0.08  # discount rate (interest rate)
    N = 30  # estimated lifetime of equipment in years

    # capital recovery factor: function of interest rate and lifetime of equipment
    CRF = (i * (1 + i) ** N) / ((1 + i) ** N - 1)

    # INDEX SETS
    # ================================================
    m.FEED = pyo.Set(initialize=stn.FEED,
                     doc='feed mixture to system')

    m.COMP = pyo.Set(initialize=stn.COMP,
                     doc="COM  ={i|i is a component in the mixture}")

    m.TASKS = pyo.Set(initialize=stn.TASKS,
                      doc="TASKS {t|t is a separation task}")

    m.STATES = pyo.Set(initialize=stn.STATES,
                       doc='STATES {s|s is a state/mixture in the system}')

    m.FS_F = pyo.Set(initialize=stn.FSf,
                     doc="FS_F = {Columns whose feed is the initial mixture}")

    m.TS_s = pyo.Set(m.STATES, initialize=stn.TSs,
                     doc='TS_s = {tasks t that the state s is able to produce}')

    m.ST_s = pyo.Set(m.STATES, initialize=stn.STs,
                     doc='ST_s = {tasks t that are able to produce state s}')

    m.ISTATE = pyo.Set(initialize=stn.ISTATE,
                       doc="ISTATE = {m | m is an intermediate state}")

    m.PRE_i = pyo.Set(m.STATES, initialize=stn.PREi,
                      doc="PRE_i = {Tasks t that produce final product i through a rectifying section}")

    m.PST_i = pyo.Set(m.STATES, initialize=stn.PSTi,
                      doc="PST_i  = {Tasks t that produce final product i through a stripping section}")

    m.RECT_s = pyo.Set(m.STATES, initialize=stn.RECTs,
                       doc="RECT_s = {Tasks t that produces state s by a rectifying section}")

    m.STRIP_s = pyo.Set(m.STATES, initialize=stn.STRIPs,
                        doc="STRIP_s {Tasks t that produces state s by a stripping section}")

    m.LK = pyo.Set(m.TASKS, initialize=stn.LK,
                   doc="light key (LK) component in a give separation task")

    m.HK = pyo.Set(m.TASKS, initialize=stn.HK,
                   doc="heavy key (HK) component in a given separation task")

    m.IREC_m = pyo.Set(m.STATES, initialize=stn.IRECs,
                       doc="IREC_m = {task t that produces intermediate state m from a rectifying section}")

    m.ISTRIP_m = pyo.Set(m.STATES, initialize=stn.ISTRIPs,
                         doc="ISTRIP_m = {task t that produces intermediate state m from a stripping section}")

    m.ROOTS = pyo.Set(initialize=stn.r, doc="Underwood roots")

    m.RUA = pyo.Set(m.TASKS, initialize=stn.RUA,
                    doc="active Underwood roots in column t")

    # CONTINUOUS POSITIVE VARIABLES
    # ================================================
    # molar flow variables
    m.FT = pyo.Var(
        m.TASKS,
        doc="Total molar flow rate entering column t [kmol/hr]",
        within=pyo.NonNegativeReals,
        bounds=(0, 20*F0),
        initialize=F0/2
    )

    m.F = pyo.Var(
        m.COMP,
        m.TASKS,
        doc="Component molar flow rate of species i entering column t [kmol/hr]",
        within=pyo.NonNegativeReals,
        bounds=(0, 20*F0),
        initialize=F0/N
    )

    m.DT = pyo.Var(
        m.TASKS,
        doc="Total distillate molar flow rate of column t [kmol/hr]",
        within=pyo.NonNegativeReals,
        bounds=(0, 20*F0),
        initialize=F0/2
    )

    m.D = pyo.Var(
        m.COMP,
        m.TASKS,
        doc="Component distillate molar flow rate of species i for column t [kmol/hr]",
        within=pyo.NonNegativeReals,
        bounds=(0, 20*F0),
        initialize=F0/N
    )

    m.BT = pyo.Var(
        m.TASKS,
        doc="Total bottoms flow rate of column t [kmol/hr]",
        within=pyo.NonNegativeReals,
        bounds=(0, 20*F0),
        initialize=F0/2
    )

    m.B = pyo.Var(
        m.COMP,
        m.TASKS,
        doc="Component bottoms flow rate of species i for column t [kmol/hr]",
        within=pyo.NonNegativeReals,
        bounds=(0, 20*F0),
        initialize=F0/N
    )

    m.Vr = pyo.Var(
        m.TASKS,
        doc="Molar flow rate of vapor in the rectifying section of column t [kmol/hr]",
        within=pyo.NonNegativeReals,
        bounds=(0, 20*F0),
        initialize=F0/(2*N)
    )

    m.Lr = pyo.Var(
        m.TASKS,
        doc="Molar flow rate of Liquid in the rectifying section of column t [kmol/hr]",
        within=pyo.NonNegativeReals,
        bounds=(0, 20*F0),
        initialize=F0/(2*N)
    )

    m.Vs = pyo.Var(
        m.TASKS,
        doc="Molar flow rate of vapor in the stripping section of column t [kmol/hr]",
        within=pyo.NonNegativeReals,
        bounds=(0, 20*F0),
        initialize=F0/(2*N)
    )

    m.Ls = pyo.Var(
        m.TASKS,
        doc="Molar flow rate of Liquid in the stripping section of column t [kmol/hr]",
        within=pyo.NonNegativeReals,
        bounds=(0, 20*F0),
        initialize=F0/(2*N)
    )
    
    # max vapor flow used in column diameter calculations
    m.V_max = pyo.Var(
        m.TASKS,
        doc="Max molar vapor flow rate for 2 column sections; V_max == max{Vr, Vs} [kmol/hr]",
        within=pyo.NonNegativeReals,
        bounds=(0, 20*F0),
        initialize=F0/(2*N)
    )

    m.rud = pyo.Var(
        m.TASKS,
        m.ROOTS,
        doc="possible active Underwood root (r) in task t",
        within=pyo.NonNegativeReals,
    )
    
    # bounding and initialization of Underwood roots based on relative volatilities
    for task in m.TASKS:
        for index, value in data.root_upper_bounds.items():
            m.rud[(task, index)].setub(value)
            
    for task in m.TASKS:
        for index, value in data.root_lower_bounds.items():
            m.rud[(task, index)].setlb(value)
        
    for task in m.TASKS:
        for index, value in data.root_initial.items():
            m.rud[(task, index)].set_value(value)

    # intermediate variable to transform to MIQCP
    m.z = pyo.Var(
        m.COMP,
        m.TASKS,
        m.ROOTS,
        doc='Intermediate variable for Underwood equations',
    )
    
    # bounding and initialization of intermediate variable (z)
    for task in m.TASKS:
        for index, value in data.z_upper_bounds.items():
            comp = index[0]
            root = index[1]
            m.z[(comp, task, root)].setub(value)

        for index, value in data.z_lower_bounds.items():
            comp = index[0]
            root = index[1]
            m.z[(comp, task, root)].setlb(value)

        for index, value in data.z_initial.items():
            comp = index[0]
            root = index[1]
            m.z[(comp, task, root)].set_value(value)

    # Column costing and sizing variables
    m.column_cost = pyo.Var(
        m.TASKS,
        doc="Capital cost of column t [$]",
        within=pyo.NonNegativeReals,
        bounds=(0, 10000000),
        initialize=1e6
    )
    m.cost_per_tray = pyo.Var(
        m.TASKS,
        doc="Capital cost of each tray for column t [$]",
        within=pyo.NonNegativeReals,
        bounds=(0, 10000000),
        initialize=1e5
    )

    m.area_final_reboiler = pyo.Var(
        m.COMP,
        doc="heat exchange area of reboiler associated with final product [m^2]",
        within=pyo.NonNegativeReals,
        bounds=(0, 100000),
        initialize=100
    )
    
    m.area_final_condenser = pyo.Var(
        m.COMP,
        doc="heat exchange area of condenser associated with final product [m^2]",
        within=pyo.NonNegativeReals,
        bounds=(0, 100000),
        initialize=100
    )

    m.area_intermediate_reboiler = pyo.Var(
        m.ISTATE,
        doc="heat exchange area of reboiler associated with intermediate product [m^2]",
        within=pyo.NonNegativeReals,
        bounds=(0, 1e6),
        initialize=100
    )

    m.area_intermediate_condenser = pyo.Var(
        m.ISTATE,
        doc="heat exchange area of condenser associated with intermediate product [m^2]",
        within=pyo.NonNegativeReals,
        bounds=(0, 1e6),
        initialize=100
    )

    m.final_condenser_cost = pyo.Var(
        m.COMP,
        doc="Capital cost of heat exchanger associated with final product i [$]",
        within=pyo.NonNegativeReals,
        bounds=(0, 1e8),
        initialize=100
    )

    m.final_reboiler_cost = pyo.Var(
        m.COMP,
        doc="Capital cost of heat exchanger associated with final product i [$]",
        within=pyo.NonNegativeReals,
        bounds=(0, 1e8),
        initialize=10000
    )

    m.inter_condenser_cost = pyo.Var(
        m.ISTATE,
        doc="Capital cost of heat exchanger associated with intermediate state s [$]",
        within=pyo.NonNegativeReals,
        bounds=(0, 1e8),
        initialize=10000
    )

    m.inter_reboiler_cost = pyo.Var(
        m.ISTATE,
        doc="Capital cost of heat exchanger associated with intermediate state s [$]",
        within=pyo.NonNegativeReals,
        bounds=(0, 1e8),
        initialize=10000
    )

    m.Ntray = pyo.Var(
        m.TASKS,
        doc="Number of trays in column t",
        within=pyo.NonNegativeReals,
        bounds=(0, 100),
        initialize=15
    )

    m.height = pyo.Var(
        m.TASKS,
        doc="height of column t [m]",
        within=pyo.NonNegativeReals,
        bounds=(0, 200),
        initialize=15,
    )

    m.Area = pyo.Var(
        m.TASKS,
        doc="Transversal area of column t [m^2]",
        within=pyo.NonNegativeReals,
        bounds=(0, 1000),
        initialize=10,
    )

    m.Diameter = pyo.Var(
        m.TASKS,
        doc="Diameter of column t [m]",
        within=pyo.NonNegativeReals,
        bounds=(0, 50),
        initialize=10,
    )

    m.Vol = pyo.Var(
        m.TASKS,
        doc="volume of column t [m^3]",
        within=pyo.NonNegativeReals,
        bounds=(0, 100000),
        initialize=100,
    )

    m.Qreb = pyo.Var(
        m.TASKS,
        doc="reboiler heat duty for column t [10^3 kJ/hr]",
        within=pyo.NonNegativeReals,
        bounds=(0, 10000000),
        initialize=1000,
    )

    m.Qcond = pyo.Var(
        m.TASKS,
        doc="condenser heat duty for column t [10^3 kJ/hr]",
        within=pyo.NonNegativeReals,
        bounds=(0, 10000000),
        initialize=10000,
    )

    m.CAPEX = pyo.Var(
        doc="Total capital expense as sum of bare module purchase prices",
        within=pyo.NonNegativeReals,
        bounds=(0, 1e10),
        initialize=10000,
    )

    m.OPEX = pyo.Var(
        doc="Total system operating expenses as sum of reboiler and condenser expenses",
        within=pyo.NonNegativeReals,
        bounds=(0, 1e10),
        initialize=10000,
    )

    # PARAMETERS
    # ================================================

    m.F0 = pyo.Param(initialize=data.F0, doc="System inlet molar flow rate [kmol/hr]")

    m.F0_comp = pyo.Param(m.COMP, initialize=Fi0, doc="System inlet molar flow rate for each species i [kmol/hr]")

    m.rec_comp = pyo.Param(m.COMP, initialize=comp_rec,
                           doc="Specified recovery for each component")

    m.alpha = pyo.Param(m.COMP, initialize=relative_volatility,
                        doc="species relative volatility")

    m.PPM = pyo.Param(initialize=sum(zf[i] * PM[i] for i in m.COMP),
                      doc="Molecular weight of feed stream")

    m.rho_V = pyo.Param(
        m.TASKS,
        doc="Density of vapor in state s",
        within=pyo.NonNegativeReals,
        initialize=m.PPM * P_abs / 0.082 / (Tf + 273))

    m.Hvap = pyo.Param(m.COMP, initialize=Hvap,
                       doc="vaporization enthalpy for each species [KJ/mol]")

    # GLOBAL CONSTRAINTS
    # ================================================

    # global constraints for initial system feed
    @m.Constraint()
    def system_feed_total_mb(m):
        "Sum of feed inputs to columns that can take initial mixture must be same as system input"
        return sum(m.FT[t] for t in m.FS_F) == F0

    @m.Constraint(m.COMP)
    def system_feed_component_mb(m, i):
        "For each component, the sum of feed inputs to columns that can take initial mixture must be same as system input"
        return sum(m.F[(i, t)] for t in m.FS_F) == m.F0_comp[i]

    # global constraints for connectivity flows between columns
    @m.Constraint(m.STATES)
    def total_mb_between_cols(m, s):
        """Constraint links the total molar outflows of distillate and bottoms of one column to the feed of another"""
        if s in m.FEED:
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
        if s in m.FEED:
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

    # global constraints for summation of components to total molar flows
    @m.Constraint(m.TASKS)
    def feed_comp_mb_global(m, t):
        """Component molar flows must sum to total molar flow for each feed stream"""
        return m.FT[t] == sum(m.F[(i, t)] for i in m.COMP)
    
    @m.Constraint(m.TASKS)
    def distillate_comp_mb_global(m, t):
        """Component molar flows must sum to total molar flow for each distillate stream"""
        return m.DT[t] == sum(m.D[(i, t)] for i in m.COMP)

    @m.Constraint(m.TASKS)
    def bottoms_comp_mb_global(m, t):
        """Component molar flows must sum to total molar flow for each bottoms stream"""
        return m.BT[t] == sum(m.B[(i, t)] for i in m.COMP)
    
    
    # DISJUNCTS FOR COLUMNS (SEPARATION TASKS)
    # ================================================
    m.column = Disjunct(m.TASKS, doc="Disjunct for column existence")
    m.no_column = Disjunct(m.TASKS, doc="Disjunct for column absence")

    # column disjunction
    @m.Disjunction(m.TASKS, doc="Column exists or does not")
    def column_no_column(m, t):
        return [m.column[t], m.no_column[t]]

    # Functions for defining mass balance and Underwood relation constraints for columns disjuncts
    # ================================================

    def _build_mass_balance_column(m: pyo.ConcreteModel, t:str, column:Disjunct) -> None:
        """Apply total and component mass balance constraints to active column disjuncts

        Args:
            m (pyo.ConcreteModel): Pyomo model object contains relevant variables, parameters
                and expressions for distillation network
            t (str): index from the set TASKS representing the column
            column (Disjunct): Pyomo Disjunct representing an active column

        Returns:
            None: function applies constraints to model object, does not return value
            
        Constraints:
            column_total_mb: Total mass balance on column
            column_component_mb: Component mass balance on column
            column_rectifying_mb: Total mass balance on rectifying section of column
            column_stripping_mb: Total mass balance on stripping section of column
            feed_comp_mb: Feed component flows sum to total feed flow
            distillate_comp_mb: Distillate component flows sum to total distillation flow
            bottoms_comp_mb: Bottom component flows sum to total bottom flow
            lk_split: Specified recovery light key component in the distillate
            hk_split: Specified recovery heavy key component in the bottoms
        """

        @column.Constraint()
        def column_total_mb(_):
            """Total mass balance on column"""
            return m.FT[t] == m.DT[t] + m.BT[t]

        @column.Constraint(m.COMP)
        def column_component_mb(_, i):
            """Component mass balance for feed and distillate in column disjunct t"""
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
        
        @column.Constraint(m.COMP)
        def lk_split(_, i):
            """Enforcing specified recovery of the light key component for the given separation task"""
            LK_alpha = m.alpha[list(m.LK[t])]
            
            if i in m.LK[t]:
                return m.D[(i, t)] >= m.rec_comp[i] * m.F[(i, t)]
            elif m.alpha[i] > LK_alpha:
                return m.D[(i, t)] == m.F[(i, t)]
            else:
                return pyo.Constraint.Skip
            
        @column.Constraint(m.COMP)
        def hk_split(_, i):
            """Enforcing specified recovery of the heavy key component for the given separation task"""
            HK_alpha = m.alpha[list(m.HK[t])]
            
            if i in m.HK[t]:
                return m.B[(i, t)] >= m.rec_comp[i] * m.F[(i, t)]
            elif m.alpha[i] < HK_alpha:
                return m.B[(i, t)] == m.F[(i, t)]
            else:
                return pyo.Constraint.Skip
            
        @column.Constraint()
        def max_vapor_flow(_):
            if pyo.value(m.Vr[t]) > pyo.value(m.Vs[t]):
                return m.V_max == m.Vr[t]
            else:
                return m.V_max[t] == m.Vs[t]

    def _build_mass_balance_no_column(m: pyo.ConcreteModel, t: str, nocolumn: Disjunct) -> None:
        """Apply total and component mass balance constraints to inactive column disjuncts.
            Set flows to zero. 

        Args:
            m (pyo.ConcreteModel): Pyomo model object contains relevant variables, parameters
                and expressions for distillation network
            t (str): index from the set TASKS representing the column
            nocolumn (Disjunct): Pyomo Disjunct representing an active column

        Returns:
            None: function applies constraints to model object, does not return value
            
        Constraints:
        All flow values set to zero
            inactive_feed_component_flow 
            inactive_distillate_component_flow
            inactive_bottoms_component_flow
            inactive_vapor_rectifying_flow
            inactive_liquid_rectifying_flow
            inactive_liquid_stripping_flow
            inactive_feed_total_flow
            inactive_distillate_total_flow
            inactive_bottoms_total_flow
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
        def inactive_vapor_stripping_flow(_):
            return m.Vs[t] == 0

        @nocolumn.Constraint()
        def inactive_max_vapor_flow(_):
            return m.V_max[t] == 0
        
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

    def _build_underwood_eqns(m: pyo.ConcreteModel, t: str, column: Disjunct) -> None:
        """Apply constraints for the Underwood equations, that act as shortcut models for column behavior, to active column disjuncts

        Args:
            m (pyo.ConcreteModel): Pyomo model object contains relevant variables, parameters
                and expressions for distillation network
            t (str): index from the set TASKS representing the column
            column (Disjunct): Pyomo Disjunct representing an active column

        Returns:
            None: function applies constraints to model object, does not return value
        
        Constraints:
            intermediate_var_con
            underwood1
            underwood2
            underwood3
        """

        # get a list of the potential active Underwood roots for a separation task t
        roots = list(m.RUA[t])

        @column.Constraint(m.COMP, roots)
        def intermediate_var_con(_, i, r):
            """Intermediate variable to reformulate Underwood equations to be bilinear"""
            return m.z[(i, t, r)] * (m.alpha[i] - m.rud[(t, r)]) == 1

        @column.Constraint(roots)
        def underwood1(_, r):
            """Underwood eqn 1 (feed equation) to determine all Underwood roots"""
            return sum((m.z[(i, t, r)] * m.alpha[i] * m.F[(i, t)]) for i in m.COMP) - (m.Vr[t] - m.Vs[t]) == 0

        @column.Constraint(roots)
        def underwood2(_, r):
            """Underwood eqn 2 (vapor equation) to determine vapor flow in rectifying section"""
            return sum((m.z[(i, t, r)] * m.alpha[i] * m.D[(i, t)]) for i in m.COMP) <= m.Vr[t]

        @column.Constraint(roots)
        def underwood3(_, r):
            """Underwood eqn 3 (vapor equation) to determine vapor flow in stripping section"""
            return -sum((m.z[(i, t, r)] * m.alpha[i] * m.B[(i, t)]) for i in m.COMP) <= m.Vs[t]

    # Functions for defining tray number, column size, and cost constraints
    # ================================================
    def _build_trays(m: pyo.ConcreteModel, t: str, column: Disjunct) -> None:
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
            """"Minimum tray number from empirical correlation. Actual number of trays as twice the minimum number"""
            # get the recovery values for the associated light and heavy key components for the task
            LK = list(m.LK[t])
            recovery_LK = m.rec_comp[LK[0]]
            
            HK = list(m.HK[t])
            recovery_HK = m.rec_comp[HK[0]]
            
            Ntray_min = pyo.log10(recovery_LK**2 / (1 - recovery_HK) ** 2) / pyo.log10(
                sum(m.alpha[i] for i in m.LK[t]) / sum(m.alpha[i] for i in m.HK[t]))
            return m.Ntray[t] == 2 * Ntray_min

    def _build_column_height(m: pyo.ConcreteModel, t: str, column: Disjunct) -> None:
        """Apply constraints for empirical relation of column height

        Args:
            m (pyo.ConcreteModel): Pyomo model object contains relevant variables, parameters
                and expressions for distillation network
            t (str): index from the set TASKS representing the column
            column (Disjunct): Pyomo Disjunct representing an active column

        Returns:
            None: function applies constraints to model object, does not return value
            
        Constraints:
            column_diameter: compute column diameter with intermediate variable in order to calculate area
            column_height: compute column height in [m]
        """
        @column.Constraint()
        def column_diameter(_):
            """Constraint for defining column diameter"""
            return m.Area[t] == pi * (m.Diameter[t] / 2)**2
        
        @column.Constraint()
        def column_height(_):
            """Empirical relation for the height of the column from Ulrich et al Ch 4. Column height in [m^2]"""
            # parameters for a quadratic fit of empirical relation for height of each tray
            fit_params = [-0.00222188,  0.08265548,  0.41101634]
            H_per_tray = m.Diameter[t]*fit_params[0]**2 + m.Diameter[t]*fit_params[1] + m.Diameter[t]*fit_params[2]
            return m.height[t] == m.Ntray[t] * H_per_tray

    def _build_column_area(m: pyo.ConcreteModel, t: str, column: Disjunct) -> None:
        """
        Function builds constraints for the column area for each column disjunct

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
            """Function calculates the area of a sieve tray column for a mixture at a fixed pressure.
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

            Calculation procedure from Chapter 9.2 of:
            Stichlmair, J. G., Klein, H., & Rehfeldt, S. (2021). Distillation Principles and Practice.
            Newark American Institute Of Chemical Engineers Ann Arbor, Michigan Proquest.
            """

            g = 9.81  # gravitational constant [m/sec^2]
            dh = dh / 1000  # unit conversion of sieve hole diameter from mm to meters
            weir_length = 0.7  # use of a relative weir length of 0.7
            V_dot = V_dot * (1 / 3600)  # conversion of units to [kmol/sec]

            # maximum gas load for large tray spacing
            F_max = 2.5 * (phi**2 * sigma * (rho_L - rho_g) * g) ** (1 / 4)

            # can use 2 criterion for minimum gas load:
            # F_min_1: non-uniform gas flow
            # F_min_2: criterion for weeping

            F_min_1 = phi * pyo.sqrt(2 * (sigma / dh))
            F_min_2 = phi * pyo.sqrt(0.37 * dh * g * ((rho_L - rho_g) ** 1.25) / rho_g**0.25)

            # take the max of the 2; using logical comparison to avoid non-differentiable max() operator
            if F_min_1 > F_min_2:
                F_min = F_min_1
            elif F_min_2 >= F_min_2:
                F_min = F_min_2

            # take the geometric mean of min and max gas load to get gas load specification for system
            F = (F_min * F_max) ** (1 / 2)

            u_g = F / pyo.sqrt(rho_g)  # calculating superficial gas velocity

            V_g = V_dot * (PM / rho_g)  # calculation of volumetric gas flow rate [m^3/sec]
            Aac = V_g / u_g  # calculation of active area of a tray [m^2]

            temp = 1 - (2 / pi) * (pyo.asin(weir_length) - pyo.sqrt(weir_length**2 - weir_length**4))
            Ac = Aac / temp  # column area [m^2]

            return Ac  # column area in [m^2]

        @column.Constraint()
        def column_area(_):
            """Calculation of column area based on empirical correlation"""
            return m.Area[t] == column_area_correlation(m.V_max[t], rho_L, m.rho_V[t], m.PPM)

    def _build_column_volume(m: pyo.ConcreteModel, t: str, column: Disjunct) -> None:
        """
        Function builds constraints for the volume of each column

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
        def column_volume(_):
            """calculation of column volume based on empirical correlation"""
            return m.Vol[t] == m.Area[t] * m.height[t]

    def _build_no_column_size(m, t, nocolumn):
        """Function builds constraints that set physical parameters of column (tray number, height,
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

    def _build_column_cost(m: pyo.ConcreteModel, t: str, column: Disjunct, no_column: Disjunct) -> None:
        """
        Function builds constraints for the cost relation for each column

        Args:
            m(pyomo.ConcreteModel): Pyomo model object contains relevant variables, parameters
                and expressions for distillation network
            t (str): index from the set TASKS representing the column
            column (pyomo.gdp.disjunct): Pyomo Disjunct representing an active column
            no_column (pyomo.gdp.disjunct): Pyomo Disjunct representing an inactive column

        Constraints:
            -column_cost: empirical correlation based on calculations in Turton et al.
                column cost as sum of cost of trays and vertical process vessel
            -no_column_cost: capital cost of column set to zero for inactive disjunct

        Return:
            None. Function adds constraints to the model but does not return a value
        """

        @column.Constraint()
        def cost_tray(_):
            # empirical cost correlation for capital cost of each tray as function of area
            return m.cost_per_tray[t] == 571.1 + 406.8 * m.Area[t] + 38 * m.Area[t] ** 2

        @column.Constraint()
        def column_cost(_):
            # values used to calculate price changes due to inflation
            CEPCI_2004 = 444.2
            CEPCI_2020 = 596.2
            inflation_ratio = CEPCI_2020 / CEPCI_2004

            Cost_trays = m.cost_per_tray[t] * m.Ntray[t]
            CP = 603.8 * m.Vol[t] + 5307
            Cost_shell = CP * (2.50 + 1.72)

            return m.column_cost[t] == (Cost_shell + Cost_trays) * inflation_ratio

        @no_column.Constraint()
        def no_column_cost(_):
            # set total cost to zero for inactive column
            return m.column_cost[t] == 0

        @no_column.Constraint()
        def no_column_tray_cost(_):
            # set the per tray cost to zero for inactive column
            return m.cost_per_tray[t] == 0

    # calling functions to build constraints for active and inactive column disjuncts
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
    m.final_heat_exchanger = Disjunct(m.COMP,
                                      doc="Disjunct for the existence of a heat exchanger associated with a final product")
    m.no_final_heat_exchanger = Disjunct(m.COMP,
                                         doc="Disjunct for the non-existence of a heat exchanger associated with a final product")

    # final product heat exchanger disjunction are indexed by pure components i in COMPS [e.g. A, B, C]
    @m.Disjunction(m.COMP, doc="Final product heat exchanger exists or does not")
    def final_heat_exchange_no_final_heat_exchange(m, i):
        return [m.final_heat_exchanger[i], m.no_final_heat_exchanger[i]]

    def _build_final_product_heat_exchanger(m: pyo.ConcreteModel, i: str,
                                            heat_exchanger: Disjunct, no_heat_exchanger: Disjunct) -> None:
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

        # Constraints for heat exchangers associated with final states produced by a rectifying section (condensers)
        if i in m.PRE_i:
            rect_tasks = m.PRE_i[i]

            @heat_exchanger.Constraint(rect_tasks)
            def condenser_heat_duty(_, t):
                """Task condenser heat duty based on enthalpy of vaporization and stripping section vapor flow rate"""
                return m.Qcond[t] * m.DT[t] == sum(m.Hvap[j] * m.D[(j, t)] * m.Vr[t] for j in m.COMP) 

            @no_heat_exchanger.Constraint(rect_tasks)
            def inactive_final_condenser(_, t):
                """final product heat exchanger for a stream from a rectifying section is not selected"""
                return m.Qcond[t] == 0

        # Constraints for heat exchangers associated with final states produced by a stripping section (reboilers)
        if i in m.PST_i:
            strip_tasks = m.PST_i[i]

            @heat_exchanger.Constraint(strip_tasks)
            def reboiler_heat_duty(_, t):
                """Task reboiler heat duty based on enthalpy of vaporization and rectifying section vapor flow rate"""
                return m.Qreb[t] * m.BT[t] == sum(m.Hvap[j] * m.B[(j, t)] * m.Vs[t] for j in m.COMP) 

            @no_heat_exchanger.Constraint(strip_tasks)
            def inactive_final_reboiler(_, t):
                """final product heat exchange for a stream from a stripping section is not selected"""
                return m.Qreb[t] == 0

    def _final_heat_exchanger_cost(m: pyo.ConcreteModel, i: str, heat_exchanger: Disjunct, no_heat_exchanger: Disjunct) -> None:
        """Function to build constraints for area and capital cost of active final product heat exchanger

        Args:
            m(pyomo.ConcreteModel): Pyomo model object contains relevant variables, parameters
                and expressions for distillation network
            i (str): index from the set COMP a final product (A, B, C, etc.)
            heat_exchanger (pyomo.gdp.disjunct): Pyomo Disjunct representing an active heat exchanger for a final product

        Constraints:
            -condenser_area: exchanger area based on empirical correlation
            -reboiler_area: exchanger area based on empirical correlation
            -condenser_cost: exchanger capital cost based on empirical correlation
            -reboiler_cost: exchanger capital cost based on empirical correlation
            -inactive_exchanger_area: area set to zero for inactive exchanger disjunct
            -inactive_reboiler_cost: capital cost set to zero for inactive exchanger disjunct
            -inactive_condenser_cost: capital cost set to zero for inactive exchanger disjunct

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

        # U for hot side of condensing steam and cold side fluid of low viscosity liquid hydrocarbons
        Ureb = 600  # [J / m^2 sec K]
        # U for hot side of hydrocarbon gases and cold side fluid of liquid water
        Ucond = 200  # [J / m^2 sec K]

        # delta log mean temp difference from condenser and reboiler
        delta_LM_T_cond = 200
        delta_LM_T_reb = 270

        # if the final state is produced by a rectifying section, will have associated condenser
        if i in m.PRE_i:

            @heat_exchanger.Constraint()
            def condenser_area(_):
                Qcon_inter = (sum(m.Qcond[t] for t in m.PRE_i[i]) * 1000)  # multiply by 1000 for unit conversion to [J/sec]
                return m.area_final_condenser[i] == Qcon_inter / (Ucond * delta_LM_T_cond)

            @heat_exchanger.Constraint()
            def condenser_cost(_):
                """Using a quadratic fit for cost correlation to get problem to be an MIQCP"""
                # parameters for a polynomial fit
                condenser_params = [-5.33104431e-02,  1.26227551e+02,  9.65399504e+03]
                return (m.final_condenser_cost[i] == (condenser_params[0] * m.area_final_condenser[i]**2
                                                                    + condenser_params[1] * m.area_final_condenser[i]
                                                                    + condenser_params[2])) 

        # if the final state is produced by a stripping section, will have associated reboiler
        if i in m.PST_i:

            @heat_exchanger.Constraint()
            def reboiler_area(_):
                Qreb_inter = (sum(m.Qreb[t] for t in m.PST_i[i]) * 1000)  # multiply by 1000 for unit conversion to [J/sec]
                return m.area_final_reboiler[i] == Qreb_inter / (Ureb * delta_LM_T_reb)

            @heat_exchanger.Constraint()
            def reboiler_cost(_):
                """Using a quadratic fit for cost correlation to get problem to be an MIQCP"""
                # parameters for a polynomial fit
                reboiler_params = [-7.10805909e-02,  1.68303402e+02,  1.28719934e+04]
                
                return (m.final_reboiler_cost[i] == (reboiler_params[0] * m.area_final_reboiler[i]**2
                                                            + reboiler_params[1] * m.area_final_reboiler[i]
                                                            + reboiler_params[2]))

        @no_heat_exchanger.Constraint()
        def inactive_reboiler_area(_):
            """For an inactive exchanger, set heat exchange area to zero"""
            return m.area_final_reboiler[i] == 0

        @no_heat_exchanger.Constraint()
        def inactive_condenser_area(_):
            """For an inactive exchanger, set heat exchange area to zero"""
            return m.area_final_condenser[i] == 0

        @no_heat_exchanger.Constraint()
        def inactive_reboiler_cost(_):
            """For an inactive reboiler, set capital cost to zero"""
            return m.final_reboiler_cost[i] == 0

        @no_heat_exchanger.Constraint()
        def inactive_condenser_cost(_):
            """For an inactive condenser, set capital cost to zero"""
            return m.final_condenser_cost[i] == 0

    # # calling functions to build constraints for active and inactive final product heat exchanger disjuncts
    for i in m.COMP:
        _build_final_product_heat_exchanger(m, i, m.final_heat_exchanger[i], m.no_final_heat_exchanger[i])
        _final_heat_exchanger_cost(m, i, m.final_heat_exchanger[i], m.no_final_heat_exchanger[i])

    # DISJUNCTS FOR INTERMEDIATE PRODUCT HEAT EXCHANGERS
    # ================================================
    m.int_heat_exchanger = Disjunct(m.ISTATE,
                                    doc="Disjunct for the existence of a heat exchanger associated with a intermediate product")
    m.no_int_heat_exchanger = Disjunct(m.ISTATE,
                                       doc="Disjunct for the non-existence of a heat exchanger associated with a intermediate product")

    # intermediate state heat exchanger disjunctions are indexed by ISTATE [e.g. AB, BC]
    @m.Disjunction(m.ISTATE,
                   doc="Intermediate product heat exchanger exists or does not")
    def int_heat_exchange_no_final_heat_exchange(m, i):
        return [m.int_heat_exchanger[i], m.no_int_heat_exchanger[i]]

    def _build_intermediate_product_heat_exchanger(m: pyo.ConcreteModel, s: str, heat_exchanger: Disjunct, no_heat_exchanger: Disjunct) -> None:
        """Function to build constraints for disjuncts of active intermediate product heat exchangers

        Args:
            m(pyomo.ConcreteModel): Pyomo model object contains relevant variables, parameters
                and expressions for distillation network
            s (str): index from the set ISTATE a intermediate product (A, B, C, etc.)
            heat_exchanger (pyomo.gdp.disjunct): Pyomo Disjunct representing an active heat exchanger for a intermediate product
            no_heat_exchanger (pyomo.gdp.disjunct): Pyomo Disjunct representing an inactive heat exchanger for a intermediate product

        Constraints:
            -int_condenser_heat_duty: Qcond is a function of species vaporization enthalpy and vapor flow rates
            -int_reboiler_heat_duty: Qreb is a function of species vaporization enthalpy and vapor flow rates
            -inactive_intermediate_condenser: set heat duty to zero for inactive disjunct
            -inactive_intermediate_reboiler: set heat duty to zero for inactive disjunct
            -inactive_heat_exchanger_mb_vapor: if there is no intermediate heat exchanger, mass balance between columns for thermal couple
            -inactive_heat_exchanger_mb_liquid: if there is no intermediate heat exchanger, mass balance between columns for thermal couple

        Return:
            None. Function adds constraints to the model but does not return a value
        """

        task_list = list(m.TS_s[s])

        @heat_exchanger.Constraint(task_list)
        def flow_constraint(_, t):
            """"If an intermediate heat exchanger is used, the feed to the next column is a saturated liquid"""
            return m.FT[t] + m.Lr[t] == m.Ls[t]

        # if intermediate state is produced by a rectifying section and it has a heat exchanger, it will be a condenser
        if s in m.IREC_m:
            rect_tasks = m.IREC_m[s]

            @heat_exchanger.Constraint(rect_tasks)
            def int_condenser_heat_duty(_, t):
                """Task reboiler heat duty based on enthalpy of vaporization and stripping section vapor flow rate"""
                return m.Qcond[t] * m.DT[t] == sum(m.Hvap[i] * m.D[(i, t)] * m.Vr[t] for i in m.COMP)

            @no_heat_exchanger.Constraint(rect_tasks)
            def inactive_intermediate_condenser(_, t):
                """intermediate product heat exchange for a stream from a rectifying section is not selected"""
                return m.Qcond[t] == 0

        # if intermediate state is produced by a stripping section and it has a heat exchanger, it will be a reboiler
        if s in m.ISTRIP_m:
            strip_tasks = m.ISTRIP_m[s]

            @heat_exchanger.Constraint(strip_tasks)
            def int_reboiler_heat_duty(_, t):
                """Task condenser heat duty based on enthalpy of vaporization and rectifying section vapor flow rate"""
                return m.Qreb[t] * m.BT[t] == sum(m.Hvap[i] * m.B[(i, t)] * m.Vr[t] for i in m.COMP)

            @no_heat_exchanger.Constraint(strip_tasks)
            def inactive_intermediate_reboiler(_, t):
                """intermediate product heat exchange for a stream from a stripping section is not selected"""
                return m.Qreb[t] == 0
        
        # mass balance equations to accounts for exchange of liquid and vapor between columns when 
        # there is no intermediate product heat exchanger (columns are thermally coupled)
        
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
        def inactive_heat_exchanger_mb_vapor(_):
            return sum(m.Vr[t] - m.Vs[t] for t in m.TS_s[s]) + vapor_strip_flow == vapor_rec_flow

        @no_heat_exchanger.Constraint()
        def inactive_heat_exchanger_mb_liquid(_):
            return sum(m.Lr[t] - m.Ls[t] for t in m.TS_s[s]) + liquid_strip_flow == liquid_rec_flow

    def _build_intermediate_heat_exchanger_cost(m: pyo.ConcreteModel, s: str, heat_exchanger: Disjunct, no_heat_exchanger: Disjunct) -> None:
        """Function to build constraints for capital cost of active intermediate product heat exchanger

        Args:
            m(pyomo.ConcreteModel): Pyomo model object contains relevant variables, parameters
                and expressions for distillation network
            s (str): index from the set ISTATE an intermediate state (AB, BC, etc.)
            heat_exchanger (pyomo.gdp.disjunct): Pyomo Disjunct representing an active heat exchanger for a intermediate product

        Constraints:
            -condenser_area: exchanger area based on empirical correlation
            -reboiler_area: exchanger area based on empirical correlation
            -condenser_cost: exchanger capital cost based on empirical correlation
            -condenser_cost: exchanger capital cost based on empirical correlation
            -inactive_exchanger_area:
            -inactive_condenser_cost:
            -inactive_reboiler_cost:

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

        # U for hot side of condensing steam and cold side fluid of low viscosity liquid hydrocarbons
        Ureb = 600  # [J / m^2 sec K]
        # U for hot side of hydrocarbon gases and cold side fluid of liquid water
        Ucond = 200  # [J / m^2 sec K]

        # delta log mean temp difference from condenser and reboiler
        delta_LM_T_cond = 200
        delta_LM_T_reb = 270

        # if the intermediate state is produced by a rectifying section, will have associated condenser
        if s in m.IREC_m:

            @heat_exchanger.Constraint()
            def condenser_area(_):
                Qcon_inter = (sum(m.Qcond[t] for t in m.IREC_m[s]) * 1000)  # multiply by 1000 for unit conversion to [J/se0c]
                return m.area_intermediate_condenser[s] == Qcon_inter / (Ucond * delta_LM_T_cond)

            @heat_exchanger.Constraint()
            def condenser_cost(_):
                """Using a quadratic fit for cost correlation to get problem to be an MIQCP"""
                # parameters for a polynomial fit to an empirical cost correlation
                condenser_params = [-5.33104431e-02,  1.26227551e+02,  9.65399504e+03]
                return (m.inter_condenser_cost[s] == (condenser_params[0] * m.area_intermediate_condenser[s]**2
                                                                    + condenser_params[1] * m.area_intermediate_condenser[s]
                                                                    + condenser_params[2]))

        # if the intermediate state is produced by a stripping section, will have associated reboiler
        if s in m.ISTRIP_m:

            @heat_exchanger.Constraint()
            def reboiler_area(_):
                Qreb_inter = (sum(m.Qreb[t] for t in m.ISTRIP_m[s]) * 1000)  # multiply by 1000 for unit conversion to [J/se0c]
                return m.area_intermediate_reboiler[s] == Qreb_inter / (Ureb * delta_LM_T_reb)

            @heat_exchanger.Constraint()
            def reboiler_cost(_):
                """Using a quadratic fit for cost correlation to get problem to be an MIQCP"""
                # parameters for a polynomial fit
                reboiler_params = [-7.10805909e-02,  1.68303402e+02,  1.28719934e+04]
                return (m.inter_reboiler_cost[s] == (reboiler_params[0] * m.area_intermediate_reboiler[s]**2
                                                            + reboiler_params[1] * m.area_intermediate_reboiler[s]
                                                            + reboiler_params[2]))
        @no_heat_exchanger.Constraint()
        def inactive_reboiler_area(_):
            """For an inactive exchanger, set heat exchange area to zero"""
            return m.area_intermediate_reboiler[s] == 0

        @no_heat_exchanger.Constraint()
        def inactive_condenser_area(_):
            """For an inactive exchanger, set heat exchange area to zero"""
            return m.area_intermediate_condenser[s] == 0

        @no_heat_exchanger.Constraint()
        def inactive_condenser_cost(_):
            """For an inactive condenser, set capital cost to zero"""
            return m.inter_condenser_cost[s] == 0

        @no_heat_exchanger.Constraint()
        def inactive_reboiler_cost(_):
            """For an inactive reboiler, set capital cost to zero"""
            return m.inter_reboiler_cost[s] == 0

    # calling functions to build constraints for active and inactive intermediate product heat exchanger disjuncts
    for s in m.ISTATE:
        _build_intermediate_product_heat_exchanger(m, s, m.int_heat_exchanger[s], m.no_int_heat_exchanger[s])
        _build_intermediate_heat_exchanger_cost(m, s, m.int_heat_exchanger[s], m.no_int_heat_exchanger[s])

    # LOGICAL CONSTRAINTS
    # ================================================

    @m.Constraint(
        m.STATES,
        doc="""A given state s can give rise to at most one task: cannot split a product
                stream and send to 2 different columns""")
    def logic1(m, s):
        if s in m.TS_s:
            return sum(m.column[t].binary_indicator_var for t in m.TS_s[s]) <= 1
        else:
            return pyo.Constraint.Skip

    # Logic 2 and Logic 3: A given state can be produced by at most 2 tasks: one must
    # be from a rectifying section and one must be from a stripping section

    @m.Constraint(m.STATES,
                  doc="State generated by rectifying section")
    def logic2(m, s):
        if s in m.RECT_s:
            tasks = list(m.RECT_s[s])
            return sum(m.column[t].binary_indicator_var for t in tasks) <= 1
        else:
            return pyo.Constraint.Skip

    @m.Constraint(m.STATES,
                  doc="State generated by stripping section")
    def logic3(m, s):
        if s in m.STRIP_s:
            tasks = list(m.STRIP_s[s])
            return sum(m.column[t].binary_indicator_var for t in tasks) <= 1
        else:
            return pyo.Constraint.Skip

    @m.Constraint(m.COMP,
                  doc="All products must be produced by at least one task")
    def logic4(m, i):
        if i in m.PRE_i:
            task1 = m.PRE_i[i]
        else:
            task1 = []
        if i in m.PST_i:
            task2 = m.PST_i[i]
        else:
            task2 = []
        return (sum(m.column[t].binary_indicator_var for t in task1)
            + sum(m.column[t].binary_indicator_var for t in task2) >= 1)

    # Pure product i can only be produced by at most one rectifying section and one stripping section
    @m.Constraint(m.COMP,
                  doc="Pure product i can only be produced by at most 1 rectifying section")
    def logic5(m, i):
        if i in m.PRE_i:
            return sum(m.column[t].binary_indicator_var for t in m.PRE_i[i]) <= 1
        else:
            return pyo.Constraint.Skip

    @m.Constraint(m.COMP,
                  doc="Pure product i can only be produced by at most 1 stripping section")
    def logic6(m, i):
        if i in m.PST_i:
            return sum(m.column[t].binary_indicator_var for t in m.PST_i[i]) <= 1
        else:
            return pyo.Constraint.Skip

    @m.LogicalConstraint(m.COMP,
                         doc="""If a final product is produced by exactly one contribution, the heat exchanger associated with
                         this product must be selected""")
    def logic7(m, i):
        # getting lists of boolean variables for column disjuncts that could produce the final state i
        if i in m.PRE_i:
            bool_vars_rectifying = [m.column[t].indicator_var for t in m.PRE_i[i]]
        else:
            bool_vars_rectifying = []
        if i in m.PST_i:
            bool_vars_stripping = [m.column[t].indicator_var for t in m.PST_i[i]]
        else:
            bool_vars_stripping = []
        
        bool_vars = bool_vars_rectifying + bool_vars_stripping
        
        # if exactly one disjunct Boolean is true (task producing final product), the heat exchanger exists
        return pyo.implies(pyo.exactly(1, bool_vars), m.final_heat_exchanger[i].indicator_var)

    @m.LogicalConstraint(m.COMP,
                         doc="""If a given final state is produced by 2 tasks,
                         then there is no heat exchanger associated with that state""")
    def logic8(m, i):
        if i in m.PRE_i:
            bool_vars_rectifying = [m.column[t].indicator_var for t in m.PRE_i[i]]
        else:
            bool_vars_rectifying = []
        if i in m.PST_i:
            bool_vars_stripping = [m.column[t].indicator_var for t in m.PST_i[i]]
        else:
            bool_vars_stripping = []

        bool_vars = bool_vars_rectifying + bool_vars_stripping
        return pyo.implies(pyo.land(bool_vars), pyo.lnot(m.final_heat_exchanger[i].indicator_var))

    # Intermediate heat exchanger logic: cannot have a heat exchanger for an intermediate state if the state is not produced by a task

    @m.Constraint(m.ISTATE)
    def logic9(m, s):
        return (1 - m.int_heat_exchanger[s].binary_indicator_var + sum(m.column[t].binary_indicator_var for t in m.ST_s[s]) >= 1)

    @m.LogicalConstraint(m.STATES, m.TASKS,
                         doc="""Connectivity relations e.g. Existence of task AB/C implies A/B""")
    def logic10(m, s, t):
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
    def logic11(m, s, t):
        if s in m.TS_s and s in m.ST_s:
            tasks = list(m.TS_s[s])
            if t in tasks:
                task_list = [m.column[k].indicator_var for k in m.ST_s[s]]
                return pyo.implies(m.column[t].indicator_var, pyo.lor(task_list))
            else:
                return pyo.Constraint.Skip
        else:
            return pyo.Constraint.Skip

    # CONSTRAINTS FOR DEFINING OBJECTIVE FUNCTION
    # ================================================
    @m.Constraint()
    def capex_def(m):
        return (m.CAPEX == sum(m.column_cost[t] for t in m.TASKS) +
                sum(m.final_reboiler_cost[i] + m.final_condenser_cost[i] for i in m.COMP) +
                sum(m.inter_reboiler_cost[s] + m.inter_condenser_cost[s] for s in m.ISTATE))

    @m.Constraint()
    def opex_def(m):
        return m.OPEX == C_cw * op_time * sum(m.Qreb[t] for t in m.TASKS) + C_h * op_time * sum(m.Qcond[t] for t in m.TASKS)

    # OBJECTIVE
    # ================================================
    # multiply sum of bare module capital expenses by capital recovery factor (CRF) to get annualized cost

    # testing superstructure variations by fixing some things as true or false
    # m.column['A/BC'].indicator_var.fix(True)

    m.obj = pyo.Objective(expr= CRF * m.CAPEX + m.OPEX, sense=pyo.minimize)

    return m

def solve_model(model):
    """Implements heuristic decomposition solution for network model from Caballero, J. A., & Grossmann, I. E. (2004)
    
    Algorithm description:
    1. Fix all intermediate heat exchangers to an initial value (i.e. False)
    2. P1k: Solve model and extract the values of separation tasks and final product heat exchangers
    3. P2k: Fix tasks and final product heat exchanger values, unfix intermediate heat exchangers, solve model
    4. Compare objective values P2k < tolernace * P2k - 1 

    Args:
        model (pyo.ConcreteModel): input model should be the GDP Pyomo Concrete Model (before transformation)
    """

    k = 1  # initialzie iterator
    epsilon = 0.9  # some tolerance for the iterative solution

    # 1. fix the Boolean variables associated with intermediate variables heat exchanger disjuncts to be False
    for s in model.ISTATE:
        model.int_heat_exchanger[s].indicator_var.fix(False)
        model.no_int_heat_exchanger[s].indicator_var.fix(True)

    # Transformation of model and solution
    pyo.TransformationFactory('core.logical_to_linear').apply_to(model)

    mbigm = pyo.TransformationFactory('gdp.bigm')
    mbigm.apply_to(model)


    # 2. P1k: Solve model and extract out values for separation tasks and final product heat exchangers
    solver = pyo.SolverFactory('gurobi')

    # solver.options = {'NumericFocus': 2,
    #                   'nonConvex': 2}
    
    P1k_results = solver.solve(model, tee=True)
    
    # get the values for separation tasks and final product heat exchangers from solution
    separation_tasks = {t: model.column[t].indicator_var.value for t in model.TASKS}
    final_exchangers = {i: model.final_heat_exchanger[i].indicator_var.value for i in model.COMP}

    # fix the sequence of separation tasks and final product heat exchangers
    for t, value in separation_tasks.items():
        model.column[t].indicator_var.fix(value)
        
    for i, value in final_exchangers.items():
        model.final_heat_exchanger[i].indicator_var.fix(value)

    # P2k: Unfix heat exchangers and solve model
    for s in model.ISTATE:
        model.int_heat_exchanger[s].indicator_var.unfix()
        model.no_int_heat_exchanger[s].indicator_var.unfix()
    
    P2K_results = solver.solve(model, tee=True)
    
    return model, P2K_results


def add_binary_cut(model, k):
    """Function to add a binary cut for the intermediate heat exchangers of the model

    Args:
        model (pyo.ConcreteModel): _description_
        solution (_type_): _description_
        k (int): _description_

    Returns:
        _type_: _description_
    """
    constraint_name = f"binary_cut_{k}"

    # get the value of current solu
    solution = {s: model.int_heat_exchanger[s].binary_indicator_var.value for s in model.ISTATE}

    def binary_cut_rule(m):
        
        
        return sum(
            (1 - m.int_heat_exchanger[s].binary_indicator_var) if solution[s] >= 0.5 else m.int_heat_exchanger[s].binary_indicator_var
            for s in m.ISTATE) >= 1

    setattr(model, constraint_name, pyo.Constraint(rule=binary_cut_rule))