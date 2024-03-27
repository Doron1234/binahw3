IDS = ["213272644", "214422750"]
from simulator import Simulator
import random
from itertools import product


class Agent:
    def __init__(self, initial_state, player_number):
        self.ids = IDS
        self.player_number = player_number
        self.my_ships = []
        self.simulator = Simulator(initial_state)
        for ship_name, ship in initial_state['pirate_ships'].items():
            if ship['player'] == player_number:
                self.my_ships.append(ship_name)

    def act(self, state):
        raise NotImplementedError


class UCTNode:
    """
    A class for a single node. not mandatory to use but may help you.
    """
    def __init__(self, parent, new_action, old_actions, player_number):

        self.parent = parent
        self.children = []
        self.actions = old_actions + [new_action]
        self.player_number = player_number
        self.finished_expansion = False
        self.num_visits = 0
        self.sum_diffs = 0
        self.children_actions = set()

    def add_child(self, child):
        self.children.append(child)
        self.children_actions.add(child.actions[-1])

    def UCB1(self):
        if self.player_number == 1:
            if self.num_visits == 0:
                return float('inf')
            return self.sum_diffs / self.num_visits + 2 * (2 * (self.num_visits ** 2)) ** 0.5
        else:
            if self.num_visits == 2:
                return float('-inf')
            return self.sum_diffs / self.num_visits - 2 * (2 * (self.num_visits ** 2)) ** 0.5




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
        for ship_name, ship in initial_state['pirate_ships'].items():
            if ship['player'] == player_number:
                self.my_ships.append(ship_name)
            else:
                self.enemy_ships.append(ship_name)

    def selection(self, UCT_tree):
        cur_node = UCT_tree.root
        while self.finished_expansion(self.state, cur_node):
            # max or min based on player number
            if cur_node.player_number == self.player_number:
                cur_node = max(cur_node.children, key=lambda x: x.UCB1())
            else:
                cur_node = min(cur_node.children, key=lambda x: x.UCB1())
            self.simulator.act(cur_node.actions[-1], cur_node.player_number)
            self.state = self.simulator.get_state()

    def expansion(self, UCT_tree, parent_node):
        raise NotImplementedError

    def simulation(self):
        raise NotImplementedError

    def backpropagation(self, simulation_result):
        raise NotImplementedError

    def act(self, state):
        raise NotImplementedError

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
        for ship in self.my_ships:
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
                        self.player_number != state["pirate_ships"][enemy_ship_name]["player"]):
                    actions[ship].add(("plunder", ship, enemy_ship_name))
            actions[ship].add(("wait", ship))

        acts_lists = list(actions.values())
        all_actions = list(product(*acts_lists))
        print("actions:", all_actions)
        return all_actions

