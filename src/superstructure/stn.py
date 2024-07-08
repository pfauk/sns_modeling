import string
import networkx as nx
import matplotlib.pyplot as plt

class State:
    """
    class to respresent states/mixtures in a separation network.
    components in the string are orderd by decresing relative volatility
    e.g. 'ABCD'

    Attributes
        state: str
            Ordered set of characters representing compounds in the state mixture
        children: list
            list of the tasks that could be generated by the given state based on separation
            splits of consecutive key components, e.g. 'ABC' = ['A/BC', 'AB/C']
        is_final: boolean
            Boolean representing if the state if a final product state, i.e. only has single species
        is_feed: boolean
            Boolean representing if the state is the feed mixture to the overall system
    """

    def __init__(self, state, is_feed=False, is_final=False):
        if not state.isalpha():
            raise ValueError("The input string must only contain letters.")

        self.state = state
        self.children = []  # list to hold Task objects that can be generated by the State
        self.is_final = is_final

        if self.is_final and is_feed:
            raise ValueError("Feed mixure must contain at least 2 species")
        self.is_feed = is_feed  # will default to False

    def add_child_task(self, task_node):
        self.children.append(task_node)

    def __repr__(self):
        if self.is_final:
            return f"State({self.state}, final={self.is_final})"
        elif not self.is_final:
            return f"State({self.state})"


class Task:
    """
    class to represent a separation task in a separation network
    components in the string are orderd by decresing relative volatility and '/' character
    represents where in the mixture the selected split is
    e.g. 'A/BCDE'

    Attributes
        name: str
            Character string representing the separation task. Must include a '/' character
        children: list
            lsit of State objects that can be produced by the task
        dist: State
            State object representing the mixture produced in the distillate of the separation task
        bot: State
            State object representing the mixture produced in the bottoms of the separation task
    """

    def __init__(self, task):
        for char in task:
            if not ((char.isalpha()) or char == "/"):
                raise ValueError(
                    "The input string must only contain letters and a '/' character."
                )
        if task.count("/") != 1:
            raise ValueError("The input string must contain exactly one '/' character.")

        self.task = task
        self.children = []
        self.dist = None
        self.bot = None

    def add_child_state(self, node, **kwargs):
        self.children.append(node)
        if "dist" in kwargs:
            self.dist = node
        if "bottoms" in kwargs:
            self.bot = node

    def __repr__(self):
        return f"Task({self.task})"

class stn:
    """
    A class to represent a state-task network for the separation of an N-component zeotropic mixture with splits between consecutive key components.

    This class provides methods to generate and manipulate a separation network, including the creation of feed and component sets, 
    tree structures representing separation tasks, and various index sets used in the separation process.

    Attributes
    ----------
    n : int
        The number of components in the mixture.
    tree : State or None
        The root of the separation tree.
    FEED : str or None
        A string representing the initial mixture feed.
    COMP : list or None
        A list of components in the mixture.
    TASKS : list or None
        A list of tasks in the separation process.
    FSf : list or None
        A list of names of separation tasks that can have the initial mixture as their feed.
    TSs : dict or None
        A dictionary where keys are non-final states and values are lists of tasks that are children of the state.
    STs : dict or None
        A dictionary mapping states to tasks that have the state as a child.
    ISTATE : list or None
        A list of intermediate states.
    PREi : dict or None
        A dictionary mapping final states to tasks that produce them through a rectifying section.
    PSTi : dict or None
        A dictionary mapping final states to tasks that produce them through a stripping section.
    STRIPs : dict or None
        A dictionary mapping states to tasks that produce them through a stripping section.
    RECTs : dict or None
        A dictionary mapping states to tasks that produce them through a rectifying section.
    IRECs : dict or None
        A dictionary mapping intermediate states to tasks that produce them through a rectifying section.
    ISTRIPs : dict or None
        A dictionary mapping intermediate states to tasks that produce them through a stripping section.
    LK : dict or None
        A dictionary mapping tasks to their light key components.
    HK : dict or None
        A dictionary mapping tasks to their heavy key components.
    r : list or None
        A list of possible Underwood roots for the mixture.
    RUA : dict or None
        A dictionary mapping tasks to active Underwood roots.

    Methods
    -------
    _generate_initial_mixture(n):
        Create feed and components sets for a separation system.
    _generate_tree(initial_state):
        Create a separation tree for a given initial state.
    _collect_tasks(node):
        Traverse the tree and collect names of all Task objects.
    _collect_states(node):
        Traverse the tree and collect names of all State objects.
    _collect_feed_columns(node):
        Collect the names of all child Tasks from the root State node of the tree.
    _collect_TSs(node):
        Traverse the tree and collect child tasks that each state is able to produce.
    _collect_STs(node, result=None):
        Traverse the tree and collect a mapping of states to tasks that have the state as a child.
    _collect_final_states_from_distillate(node, current_tasks=None, result=None):
        Traverse the tree and collect a mapping of tasks that produce a final state from a rectifying section.
    _collect_final_states_from_bottoms(node, current_tasks=None, result=None):
        Traverse the tree and collect a mapping of tasks that produce a final state from a stripping section.
    _collect_states_from_bottoms(node, current_tasks=None, result=None):
        Traverse the tree and collect a mapping of tasks that produce a state from a stripping section.
    _collect_states_from_distillate(node, current_tasks=None, result=None):
        Traverse the tree and collect a mapping of tasks that produce a state from a rectifying section.
    _generarte_intermediate_rectifying(rect, istates):
        Produce a mapping of tasks that produce an intermediate state through a rectifying section.
    _generarte_intermediate_stripping(strip, istates):
        Produce a mapping of tasks that produce an intermediate state through a stripping section.
    _generate_light_key(tasks):
        Generate a dictionary of tasks with the values being the light key component in a given separation.
    _generate_heavy_key(tasks):
        Generate a dictionary of tasks with the values being the heavy key component in a given separation.
    _generate_underwood_roots(n):
        Generate possible Underwood roots for a mixture with N components.
    _generate_active_roots(tasks, roots):
        Generate a dictionary mapping tasks to their active Underwood roots.
    generate_tree():
        Generate the separation tree and initial mixture components.
    generate_index_sets():
        Generate and populate various index sets used in the separation process.
    _print_tree(node, level=0):
        Recursively print the tree structure.
    print_tree():
        Print the entire separation tree.
    add_nodes_edges(graph, node, pos=None, x=0, y=0, layer=1, parent=None):
        Add nodes and edges to a graph for visualization.
    display_tree():
        Display the separation tree using networkx and matplotlib.
    print_sets():
        Print the generated index sets.
    """

    def __init__(self, n):
        self.n = n
        self.tree = None
        self.FEED = None
        self.COMP = None
        self.TASKS = None
        self.COMP = None
        self.FSf = None
        self.TSs = None
        self.STs = None
        self.ISTATE = None
        self.PREi = None
        self.PSTi = None
        self.STRIPs = None
        self.RECTs = None
        self.IRECs = None
        self.ISTRIPs = None
        self.LK = None
        self.HK = None
        self.r = None
        self.RUA = None

    def _generate_initial_mixture(self, n):
        """
        Create feed and components sets for a separation system

        Takes the number of components in a mixture as inputs and returns a string
        of ordered capital letters representing the components in the mixture and the same letters, separated, in a list

        e.g. n = 6, feed_mixture = 'ABCDEF', components = ['A', 'B', 'C', 'D', 'E', 'F']
        """
        if not isinstance(n, (int, float)) and not n == int(n):
            raise TypeError("n must be an integer")

        if n <= 0 or n > 26:
            raise ValueError("Input must be a positive integer between 1 and 26.")

        alphabet = string.ascii_uppercase
        feed_mixture = alphabet[:n]
        components = list(feed_mixture)

        return feed_mixture, components

    def _generate_tree(self, initial_state):
        """"
        Create separation tree for a given initial state

        Takes an initial state ('ABCD') and recursively builds a tree of alternating state and task nodes.
        Returns the root node in the tree.
        """
        if len(initial_state) == 1:
            return State(initial_state, is_final=True)

        state_node = State(initial_state, is_final=False)
        for i in range(1, len(initial_state)):
            distillate = initial_state[:i]
            bottoms = initial_state[i:]

            task_node = Task(f"{distillate}/{bottoms}")

            distillate_state = self._generate_tree(distillate)
            bottoms_state = self._generate_tree(bottoms)

            task_node.add_child_state(distillate_state, dist=True)
            task_node.add_child_state(bottoms_state, bottoms=True)

            state_node.add_child_task(task_node)

        return state_node

    def _collect_tasks(self, node):
        """
        Traverse the tree and collect names of all Task objects.

        Args
        ----------
        node : State
            The root node of the tree or subtree

        Returns
        -------
        unique_task_names : list
            A list of names of all Task objects
        """
        task_names = []

        if isinstance(node, Task):
            task_names.append(node.task)

        for child in node.children:
            task_names.extend(self._collect_tasks(child))

        unique_task_names = sorted(list(set(task_names)))  # remove duplicates and sort
        return unique_task_names

    def _collect_states(self, node):
        """
        Traverse the tree and collect names of all State objects.

        Args:
        ----------
        node : State
            The root node of the tree or subtree

        Returns
        -------
        unique_state_names : list
            A list of names of all State objects
        """
        state_names = []

        if isinstance(node, State):
            state_names.append(node.state)

        for child in node.children:
            state_names.extend(self._collect_states(child))

        unique_state_names = sorted(list(set(state_names)))  # remove duplicates and sort

        return unique_state_names

    def _collect_feed_columns(self, node):
        """
        Collect the names of all child Tasks from the root State node of the tree

        Args:
        ----------
        node : State
            The root node of the tree or subtree

        Returns
        -------
        FSf : list
            A list of names of all separation tasks that can have the initial mixture
            as their feed
        """
        FSf = []

        for t in node.children:
            FSf.append(t.task)

        return FSf

    def _collect_TSs(self, node):
        """
        Traverse the tree and collect child tasks that each state is able to produce

        Builds the index set TS_s = {tasks t that the state s is able to produce}
        e.g. TS_{ABC} = (A/BC, AB/C)

        Args
        ----------
        node : State
            The root node of the tree or subtree

        Returns
        -------
        state_tasks : dict
            dictionary where keys are non-final states and values are lists of tasks that are children of the state
        """
        state_tasks = {}

        if isinstance(node, State) and not node.is_final:
            state_tasks[node.state] = tuple([child.task for child in node.children if isinstance(child, Task)])

        for child in node.children:
            child_state_tasks = self._collect_TSs(child)
            state_tasks.update(child_state_tasks)

        return state_tasks

    def _collect_STs(self, node, result=None):
        """
        Traverse the tree and collect a mapping of states to tasks that have the state as a child

        Builds the index set ST_s = {tasks t that are able to produce state s}
        e.g. ST_{C} = (AB/C, B/C)

        Args
        ----------
        node : State
            The current node in the tree

        result : dict
            The dictionary to store the mapping. Defaults to None.

        Returns
        -------
        result : dict
            A dictionary with states as keys and lists of tasks that produce that state as values
        """
        if result is None:
            result = {}  # create initial empty dictionary to store result

        if isinstance(node, State):
            for task in node.children:
                if task.dist.state not in result:
                    result[task.dist.state] = []
                result[task.dist.state].append(task.task)

                if task.bot.state not in result:
                    result[task.bot.state] = []
                result[task.bot.state].append(task.task)

                self._collect_STs(task, result)
        elif isinstance(node, Task):
            self._collect_STs(node.dist, result)
            self._collect_STs(node.bot, result)

        return result

    def _collect_final_states_from_distillate(self, node, current_tasks=None, result=None):
        """
        Traverse the tree and collect a mapping of tasks that produce a final state from a rectifying section

        Builds the index set PRE_i = {Tasks t that produce final product i through a rectifying section}
        e.g. PRE_{A} = (A/BC, A/B)

        Args
        ----------
        node : State
            The current node in the tree
        current_tasks : list
            List to store the values of tasks that produce a given final state i
        result : dict
            The dictionary to store the mapping. Defaults to None.

        Returns
        -------
        result : dict
            A dictionary with final states as keys and lists of tasks that produce that state from a rectifying section as values
        """
        if current_tasks is None:
            current_tasks = []
        if result is None:
            result = {}

        if isinstance(node, State):
            for task in node.children:
                self._collect_final_states_from_distillate(task, current_tasks, result)
        elif isinstance(node, Task):
            # check if the current node is a Task that produces a final State node in the distillate
            if node.dist and node.dist.is_final:
                if node.dist.state not in result:
                    result[node.dist.state] = []
                result[node.dist.state].append(node.task)
            self._collect_final_states_from_distillate(node.dist, current_tasks + [node.task], result)
            self._collect_final_states_from_distillate(node.bot, current_tasks, result)

        return result

    def _collect_final_states_from_bottoms(self, node, current_tasks=None, result=None):
        """
        Traverse the tree and collect a mapping of tasks that produce a final state from a stripping section

        Builds the index set PST_i = {Tasks t that produce final product i through a stripping section}
        e.g. PST_{B} = (A/B)

        Args
        ----------
        node : State
            The current node in the tree
        current_tasks : list
            List to store the values of tasks that produce a given final state i
        result : dict
            The dictionary to store the mapping. Defaults to None.

        Returns
        -------
        result : dict
            A dictionary with final states as keys and lists of tasks that produce that state from a stripping section as values
        """
        if current_tasks is None:
            current_tasks = []
        if result is None:
            result = {}

        if isinstance(node, State):
            for task in node.children:
                self._collect_final_states_from_bottoms(task, current_tasks, result)
        elif isinstance(node, Task):
            # check if the current node is a Task that produces a final State node in the bottoms
            if node.bot and node.bot.is_final:
                if node.bot.state not in result:
                    result[node.bot.state] = []
                result[node.bot.state].append(node.task)
            self._collect_final_states_from_bottoms(node.bot, current_tasks + [node.task], result)
            self._collect_final_states_from_bottoms(node.dist, current_tasks, result)

        return result

    def _collect_states_from_bottoms(self, node, current_tasks=None, result=None):
        """
        Traverse the tree and collect a mapping of tasks that are produce a state from a stripping section

        Builds the index set STRIP_s ={taks t that produces state s by a stripping section}
        e.g. STRIP_{BC} = (A/BC)

        Args
        ----------
        node : State
            The current node in the tree
        current_tasks : list
            List to store the values of tasks that produce a given state s
        result : dict
            The dictionary to store the mapping. Defaults to None.

        Returns
        -------
        result : dict
            A dictionary with states as keys and lists of tasks that produce that state from a stripping section as values
        """
        if current_tasks is None:
            current_tasks = []
        if result is None:
            result = {}

        if isinstance(node, State):
            for task in node.children:
                self._collect_states_from_bottoms(task, current_tasks, result)
        elif isinstance(node, Task):
            # check if the current node is a Task that produces a State node in the bottoms
            if node.bot:
                if node.bot.state not in result:
                    result[node.bot.state] = []
                result[node.bot.state].append(node.task)
            self._collect_states_from_bottoms(node.bot, current_tasks + [node.task], result)
            self._collect_states_from_bottoms(node.dist, current_tasks, result)

        return result

    def _collect_states_from_distillate(self, node, current_tasks=None, result=None):
        """
        Traverse the tree and collect a mapping of tasks that are produce a state from a rectifying section

        Builds the index set RECT_s = {taks t that produces state s by a rectifying section}
        e.g. RECT_{AB} = (AB/C)

        Args
        ----------
        node : State
            The current node in the tree
        current_tasks : list
            List to store the values of tasks that produce a given state s
        result : dict
            The dictionary to store the mapping. Defaults to None.

        Returns
        -------
        result : dict
            A dictionary with states as keys and lists of tasks that produce that state from a rectifying section as values
        """

        if current_tasks is None:
            current_tasks = []
        if result is None:
            result = {}

        if isinstance(node, State):
            for task in node.children:
                self._collect_states_from_distillate(task, current_tasks, result)
        elif isinstance(node, Task):
            # check if the current node is a Task that produces a State node in the distillate
            if node.dist:
                if node.dist.state not in result:
                    result[node.dist.state] = []
                result[node.dist.state].append(node.task)
            self._collect_states_from_distillate(node.dist, current_tasks + [node.task], result)
            self._collect_states_from_distillate(node.bot, current_tasks, result)

        return result

    def _generarte_intermediate_rectifying(self, rect, istates):
        """
        produce a mapping of tasks that produce an intermediate state through a rectifying section

        Builds the index set IREC_m = {task t that produces intermediate state m from a rectifying section}
        e.g. IREC_{AB} = (AB/C)

        Args
        ----------
        rect : dict
            Dictionary with mapping of all tasks that produce any state through a rectifying section
        istates : list
            list of intermediate states

        Returns
        -------
        result : dict
            A dictionary with intermediate states as keys and lists of tasks that produce that state from a rectifying section as values
        """
        irect = rect.copy()
        return {key: irect[key] for key in istates if key in irect}

    def _generarte_intermediate_stripping(self, strip, istates):
        """
        produce a mapping of tasks that produce an intermediate state through a stripping section

        Builds the index set ISTRIP_m = {task t that produces intermediate state m from a stripping section}
        e.g. ITRIP_{BC} = (A/BC)

        Args
        ----------
        rect : dict
            Dictionary with mapping of all tasks that produce any state through a stripping section
        istates : list
            list of intermediate states

        Returns
        -------
        result : dict
            A dictionary with intermediate states as keys and lists of tasks that produce that state from a stripping section as values
        """
        istrip = strip.copy()
        return {key: istrip[key] for key in istates if key in istrip}

    def _generate_light_key(self, tasks):
        """
        generates a dicitionary of tasks with the values being the light key component in a given separation

        Builds the index set LK_t = {i is the light key component in separation task t}
        e.g. LK_{AB/C} = B

        Args
        ----------
        tasks : list
            list of all separation tasks in a network

        Returns
        -------
        dict
            dictionary with tasks as keys and single species as values.
        """
        result = {}
        for task in tasks:
            if '/' in task:
                split_index = task.index('/')
                if split_index > 0:
                    result[task] = task[split_index - 1]
                else:
                    result[task] = None  # Handle edge case where '/' is the first character
            else:
                result[task] = None  # Handle edge case where no '/' is present

        return result

    def _generate_heavy_key(self, tasks):
        """
        generates a dicitionary of tasks with the values being the heavy key component in a given separation

        Builds the index set HK_t = {i is the heavy key component in separation task t}
        e.g. HK_{AB/C} = C

        Args
        ----------
        tasks : list
            list of all separation tasks in a network

        Returns
        -------
        dict
            dictionary with tasks as keys and single species as values.
        """
        result = {}
        for task in tasks:
            if '/' in task:
                split_index = task.index('/')
                if split_index > 0:
                    result[task] = task[split_index + 1]
                else:
                    result[task] = None  # Handle edge case where '/' is the first character
            else:
                result[task] = None  # Handle edge case where no '/' is present

        return result

    def _generate_underwood_roots(self, n):
        """Function to generate possible Underwood roots for a mixture with
        N components. Will have N-1 possible active roots"""
        roots = []
        for i in range(1, n):
            temp = "r" + str(i)
            roots.append(temp)

        return roots

    def _generate_active_roots(self, tasks, roots):
        task_roots = {}

        # Create a dictionary that maps upper case letters to array indices
        species = string.ascii_uppercase
        sepecies_to_index = {letter: index for index, letter in enumerate(species)}

        for task in tasks:
            # Calculate the number of letters excluding the '/' to determine the number of splits
            num_splits = len(task.replace('/', '')) - 1
            light_species = task[0]
            light_index = sepecies_to_index[light_species]

            # Assign the corresponding roots to the task
            task_roots[task] = tuple(roots[light_index:light_index + num_splits])
        return task_roots

    def generate_tree(self):
        self.FEED, self.COMP = self._generate_initial_mixture(self.n)
        self.tree = self._generate_tree(self.FEED)

    def generate_index_sets(self):
        self.FEED = [self.FEED]
        self.TASKS = self._collect_tasks(self.tree)
        self.STATES = self._collect_states(self.tree)
        self.FSf = self._collect_feed_columns(self.tree)
        self.TSs = self._collect_TSs(self.tree)

        self.STs = self._collect_STs(self.tree)
        # removing duplicates, making values tuples, and sorting keys alphabetically
        for key, vals in self.STs.items():
            self.STs[key] = tuple(set(vals))

        self.ISTATE = [s for s in self.STATES if s not in self.COMP and s not in self.FEED]

        self.PREi = self._collect_final_states_from_distillate(self.tree)
        # removing duplicates, making values tuples, and sorting keys alphabetically
        for key, vals in self.PREi.items():
            self.PREi[key] = tuple(set(vals))

        self.PREi = {key: self.PREi[key] for key in sorted(self.PREi)}

        self.PSTi = self._collect_final_states_from_bottoms(self.tree)
        # removing duplicates, making values tuples, and sorting keys alphabetically
        for key, vals in self.PSTi.items():
            self.PSTi[key] = tuple(set(vals))

        self.PSTi = {key: self.PSTi[key] for key in sorted(self.PSTi)}

        self.STRIPs = self._collect_states_from_bottoms(self.tree)
        for key, vals in self.STRIPs.items():
            self.STRIPs[key] = tuple(set(vals))

        self.RECTs = self._collect_states_from_distillate(self.tree)
        for key, vals in self.RECTs.items():
            self.RECTs[key] = tuple(set(vals))

        self.IRECs = self._generarte_intermediate_rectifying(self.RECTs, self.ISTATE)
        self.ISTRIPs = self._generarte_intermediate_stripping(self.STRIPs, self.ISTATE)

        self.LK = self._generate_light_key(self.TASKS)
        self.HK = self._generate_heavy_key(self.TASKS)
        self.r = self._generate_underwood_roots(self.n)
        self.RUA = self._generate_active_roots(self.TASKS, self.r)

    # class methods for visualization
    def _print_tree(self, node, level=0):
        indent = "  " * level
        print(f"{indent}{node}")
        if isinstance(node, State):
            for child in node.children:
                self._print_tree(child, level + 1)
        elif isinstance(node, Task):
            for child in node.children:
                self._print_tree(child, level + 1)

    def print_tree(self):
        self._print_tree(self.tree)

    def add_nodes_edges(self, graph, node, pos=None, x=0, y=0, layer=1, parent=None):
        if pos is None:
            pos = {}
        if isinstance(node, State):
            graph.add_node(node.state, shape='o', color='lightblue', style='filled')
            pos[node.state] = (x, y)
            if parent:
                graph.add_edge(parent, node.state)
            for i, child in enumerate(node.children):
                self.add_nodes_edges(graph, child, pos, x + (i - len(node.children) / 2) * (1 / layer), y - 1, layer + 1, node.state)
        elif isinstance(node, Task):
            graph.add_node(node.task, shape='s', color='lightgreen', style='filled')
            pos[node.task] = (x, y)
            if parent:
                graph.add_edge(parent, node.task)
            for i, child in enumerate(node.children):
                self.add_nodes_edges(graph, child, pos, x + (i - len(node.children) / 2) * (1 / layer), y - 1, layer + 1, node.task)
        return pos

    def display_tree(self):
        tree = self.tree
        graph = nx.DiGraph()
        pos = self.add_nodes_edges(graph, tree)

        shapes = set((aShape[1]["shape"] for aShape in graph.nodes(data=True)))

        for shape in shapes:
            shape_nodes = [sNode[0] for sNode in filter(lambda x: x[1]["shape"] == shape, graph.nodes(data=True))]
            shape_colors = [graph.nodes[n]['color'] for n in shape_nodes]
            nx.draw_networkx_nodes(graph, pos, node_shape=shape, nodelist=shape_nodes, node_color=shape_colors, node_size=450)

        nx.draw_networkx_edges(graph, pos)
        nx.draw_networkx_labels(graph, pos, font_size=8)

        plt.show()

    def print_sets(self):
        print()
        print(f"Number of components: {self.n}")
        print(f"Number of tasks: {len(self.TASKS)}")
        print(f'Number of states: {len(self.STATES)}')
        print(f'1. FEED: {self.FEED}')
        print(f'2. COMP: {self.COMP}')
        print(f'3. TASKS: {self.TASKS}')
        print(f'4. STATES: {self.STATES}')
        print(f'5. FS_f: {self.FSf}')
        print(f'6. TS_s: {self.TSs}')
        print(f'7. STs: {self.STs}')
        print(f'8. ISTATE: {self.ISTATE}')
        print(f'9. PRE_i: {self.PREi}')
        print(f'10. PST_i: {self.PSTi}')
        print(f'11. STRIPs: {self.STRIPs}')
        print(f'12. RECTs: {self.RECTs}')
        print(f'13. LK: {self.LK}')
        print(f'14. HK: {self.HK}')
        print(f'15. IRECs: {self.IRECs}')
        print(f'16. ISTRIPs: {self.ISTRIPs}')
        print(f'17. r: {self.r}')
        print(f'18. RUA: {self.RUA}')

if __name__ == "__main__":

    n = 6  # specify the number of components in the feed mixture
    network = stn(n)
    network.generate_tree()
    network.print_tree()
    network.generate_index_sets()
    network.print_sets()
    network.display_tree()
