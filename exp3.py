import math

IDS = ["213272644", "214422750"]
from simulator import Simulator
import random
from itertools import product


class Agent:
    def __init__(self, initial_state, player_number):
        self.uct = UCTAgent(initial_state, player_number)
        self.ids = self.uct.ids

    def act(self, state):
        return self.uct.act(state)


class UCTNode:
    """
    A class for a single node. not mandatory to use but may help you.
    """

    def __init__(self, parent, new_action, old_actions, player_number):

        self.parent = parent
        self.children = []
        if new_action is None:
            self.actions = old_actions
        else:
            self.actions = old_actions + [new_action]
        self.player_number = player_number
        self.num_visits = 0
        self.sum_diffs = 0
        self.children_actions = set()

    def add_child(self, child):
        self.children.append(child)
        self.children_actions.add(child.actions[-1])

    def get_num_visits(self):
        return self.num_visits

    def update(self, result):
        self.num_visits += 1
        self.sum_diffs += result

    def UCB1(self):
        if self.player_number == 1:
            if self.num_visits == 0:
                return float('inf')
            return self.sum_diffs / self.num_visits + (
                    2 * (math.log(self.parent.get_num_visits())) / self.num_visits) ** 0.5
        else:
            if self.num_visits == 2:
                return float('-inf')
            return self.sum_diffs / self.num_visits + (
                    2 * (math.log(self.parent.get_num_visits())) / self.num_visits) ** 0.5

    def get_empirical_mean(self):
        return self.sum_diffs / self.num_visits


class UCTTree:
    """
    A class for a Tree. not mandatory to use but may help you.
    """

    def __init__(self, root):
        self.root = root


class UCTAgent:
    def __init__(self, initial_state, player_number):
        self.ids = IDS
        self.initial_state = initial_state
        self.player_number = player_number
        self.pirate_ships = initial_state['pirate_ships']
        self.my_ships = []
        self.enemy_ships = []
        self.simulator = Simulator(initial_state)
        self.current_player = player_number
        for ship_name, ship in initial_state['pirate_ships'].items():
            if ship['player'] == player_number:
                self.my_ships.append(ship_name)
            else:
                self.enemy_ships.append(ship_name)

    def selection(self, UCT_tree):
        cur_node = UCT_tree.root
        self.state = self.simulator.get_state()

        while self.finished_expansion(self.state, cur_node):
            # max or min based on player number
            if cur_node.player_number == self.player_number:
                cur_node = max(cur_node.children, key=lambda x: x.UCB1())
            else:
                cur_node = min(cur_node.children, key=lambda x: x.UCB1())
            self.simulator.act(cur_node.actions[-1], cur_node.player_number)
            self.state = self.simulator.get_state()
        return cur_node

    def expansion(self, UCT_tree, parent_node):
        self.state = self.simulator.get_state()
        actions = self.possible_actions(self.state)
        new_actions = actions - parent_node.children_actions
        random_action = random.choice(list(new_actions))
        new_node = UCTNode(parent_node, random_action, parent_node.actions, 3 - parent_node.player_number)
        parent_node.add_child(new_node)
        self.simulator.act(new_node.actions[-1], new_node.player_number)
        self.state = self.simulator.get_state()
        self.last_node = new_node

    def simulation(self):
        # while self.simulator.turns_to_go > 0:
        counter = 0
        while counter < 20:
            counter += 1
            cur_state = self.state
            if self.current_player == 1:
                order = (1, 2)
                for player in order:
                    self.current_player = player
                    actions = self.possible_actions(cur_state)
                    random_action = random.choice(list(actions))
                    self.simulator.act(random_action, player)
                    cur_state = self.simulator.get_state()
                self.simulator.check_collision_with_marines()
                self.simulator.move_marines()
                self.current_player = 1
            else:
                actions = self.possible_actions(cur_state)
                random_action = random.choice(list(actions))
                self.simulator.act(random_action, self.current_player)
                cur_state = self.simulator.get_state()
                self.simulator.check_collision_with_marines()
                self.simulator.move_marines()
                self.current_player = 1
                actions = self.possible_actions(cur_state)
                random_action = random.choice(list(actions))
                self.simulator.act(random_action, self.current_player)
                cur_state = self.simulator.get_state()
                self.current_player = 2

        self.simulator.set_state(self.initial_state)
        self.simulator.turns_to_go = self.initial_state["turns to go"]
        ret = self.simulator.get_score()[f"player {self.player_number}"] - self.simulator.get_score()[f"player {3 - self.player_number}"]
        self.simulator.score = {'player 1': 0, 'player 2': 0}
        return ret

    def backpropagation(self, simulation_result):
        node = self.last_node
        while node is not None:
            node.update(simulation_result)
            node = node.parent

    def act(self, state):
        self.state = state
        root = UCTNode(None, None, [], 3 - self.player_number)
        tree = UCTTree(root)
        for i in range(200):
            self.simulator = Simulator(state)
            cur_node = self.selection(tree)
            self.expansion(tree, cur_node)
            simulation_result = self.simulation()
            self.backpropagation(simulation_result)
        best_node = max(tree.root.children, key=lambda x: x.get_empirical_mean())
        return best_node.actions[0]

    def finished_expansion(self, state, node):
        actions = self.possible_actions(state)
        diff = actions - node.children_actions
        if len(diff) == 0:
            return True
        return False

    def possible_actions(self, state):
        # 5 actions - "sail", “collect_treasure”, “deposit_treasures”, "plunder",
        # state =
        # map : map 2d array
        # base: tuple of base loc
        # pirate ships: dict of pirate ships name and their location, capacity, player
        # treasures: dict of treasures names and their location, reward
        # marine ships: dict of marine ships name and their index, path
        # turns to go: int
        actions = {}
        self.simulator.set_state(state)
        collected_treasures = []
        relevant_ships = self.my_ships if self.current_player == self.player_number else self.enemy_ships
        for ship in relevant_ships:
            actions[ship] = set()
            neighboring_tiles = self.simulator.neighbors(state["pirate_ships"][ship]["location"])
            for tile in neighboring_tiles:
                actions[ship].add(("sail", ship, tile))
            if state["pirate_ships"][ship]["capacity"] > 0:
                for treasure in state["treasures"].keys():
                    if state["pirate_ships"][ship]["location"] in self.simulator.neighbors(
                            state["treasures"][treasure]["location"]) and treasure not in collected_treasures:
                        actions[ship].add(("collect", ship, treasure))
                        collected_treasures.append(treasure)
            for treasure in state["treasures"].keys():
                if (state["pirate_ships"][ship]["location"] == state["base"]
                        and state["treasures"][treasure]["location"] == ship):
                    actions[ship].add(("deposit", ship, treasure))
            for enemy_ship_name in state["pirate_ships"].keys():
                if (state["pirate_ships"][ship]["location"] == state["pirate_ships"][enemy_ship_name]["location"] and
                        self.current_player != state["pirate_ships"][enemy_ship_name]["player"]):
                    actions[ship].add(("plunder", ship, enemy_ship_name))
            actions[ship].add(("wait", ship))

        acts_lists = list(actions.values())
        all_actions = set(product(*acts_lists))
        # print("actions:", all_actions)
        return all_actions
