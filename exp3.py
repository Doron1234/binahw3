IDS = ["213272644", "214422750"]

from simulator import Simulator
import random
from itertools import product
import math

random.seed(42)


# class Agent:
#     def __init__(self, initial_state, player_number):
#         self.ids = IDS
#         self.player_number = player_number
#         self.my_ships = []
#         self.simulator = Simulator(initial_state)
#         for ship_name, ship in initial_state['pirate_ships'].items():
#             if ship['player'] == player_number:
#                 self.my_ships.append(ship_name)
#
#     def act(self, state):
#         raise NotImplementedError


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

    def __init__(self, parent, action, player_number):

        self.parent = parent
        self.children = []
        self.action = action
        self.player_number = player_number
        self.num_visits = 0
        self.sum_diffs = 0
        self.children_actions = set()

    def add_child(self, action):
        if action not in self.children_actions:
            child = UCTNode(self, action, 3 - self.player_number)
            self.children.append(child)
            self.children_actions.add(action)

    def select_child(self, actions, h=None):
        relevant_children = [child for child in self.children if child.action in actions]
        if not relevant_children:
            return None
        return max(relevant_children, key=lambda child: child.UCB1(h))

    def update(self, result):
        self.num_visits += 1
        self.sum_diffs += result

    def UCB1(self, h=None):
        if self.num_visits == 0:
            return float('inf')
        is_p1 = 1 if self.player_number == 1 else -1
        if h is None:
            # normal case for p1 but if p2 it is the opposite of what it minimizes so it maxes it as well
            return self.get_empirical_mean() + is_p1 * (
                    2 * (math.log(self.parent.num_visits)) / self.num_visits) ** 0.5
        else:
            return self.get_empirical_mean() + is_p1 * ((
                                                                2 * (math.log(
                                                            self.parent.num_visits)) / self.num_visits) ** 0.5 + h(
                self.action))

    def get_empirical_mean(self):
        return self.sum_diffs / self.num_visits


class UCTTree:
    """
    A class for a Tree. not mandatory to use but may help you.
    """

    def __init__(self):
        raise NotImplementedError


class UCTAgent:
    def __init__(self, initial_state, player_number):
        self.ids = IDS
        self.player_number = player_number
        self.my_ships = []
        self.enemy_ships = []
        self.simulator = Simulator(initial_state)
        for ship_name, ship in initial_state['pirate_ships'].items():
            if ship['player'] == player_number:
                self.my_ships.append(ship_name)
            else:
                self.enemy_ships.append(ship_name)
        self.current_player = player_number
        self.root = UCTNode(None, None, 3 - player_number)
        self.state_for_h = None
        # self.expansion(self.root)

    # def selection(self, UCT_tree):
    def selection(self):
        possible_actions = self.possible_actions(self.simulator.state)
        cur_node = self.root.select_child(possible_actions, self.h)
        if cur_node is None:
            return self.root
        while True:
            if self.current_player == 1:
                self.simulator.act(cur_node.action, self.current_player)
                self.current_player = 2
            else:
                self.simulator.act(cur_node.action, self.current_player)
                self.current_player = 1
                self.simulator.check_collision_with_marines()
                self.simulator.move_marines()
            possible_actions = self.possible_actions(self.simulator.state)
            next_node = cur_node.select_child(possible_actions)
            if next_node is not None:
                cur_node = next_node
            else:
                return cur_node

    # def expansion(self, UCT_tree, parent_node):
    def expansion(self, parent_node):
        # should be here after the action in parent_node was executed
        all_actions = self.possible_actions(self.simulator.state)
        for action in all_actions:
            parent_node.add_child(action)

    def simulation(self):
        counter = 0
        while counter < 50:
            counter += 1
            cur_state = self.simulator.state
            if self.current_player == 1:
                pos_actions = list(self.possible_actions(cur_state))
                h_scores = [1.25**(self.h(action, cur_state) + 1) for action in pos_actions]
                action = random.choices(pos_actions, weights=h_scores)[0]
                # action = random.choice(list(self.possible_actions(cur_state)))
                self.simulator.act(action, self.current_player)
                cur_state = self.simulator.state
                self.current_player = 2
                action = random.choice(list(self.possible_actions(cur_state)))
                self.simulator.act(action, self.current_player)
                self.simulator.check_collision_with_marines()
                self.simulator.move_marines()
                self.current_player = 1
            else:
                pos_actions = list(self.possible_actions(cur_state))
                h_scores = [1.25**(self.h(action, cur_state) + 1) for action in pos_actions]
                action = random.choices(pos_actions, weights=h_scores)[0]
                # action = random.choice(list(self.possible_actions(cur_state)))
                self.simulator.act(action, self.current_player)
                cur_state = self.simulator.state
                self.simulator.check_collision_with_marines()
                self.simulator.move_marines()
                self.current_player = 1
                action = random.choice(list(self.possible_actions(cur_state)))
                self.simulator.act(action, self.current_player)
                self.current_player = 2
        return self.simulator.get_score()[f"player {self.player_number}"] \
            - self.simulator.get_score()[f"player {3 - self.player_number}"]

    def backpropagation(self, simulation_result, node):
        while node is not None:
            node.update(simulation_result)
            node = node.parent

    def act(self, state):
        self.root = UCTNode(None, None, 3 - self.player_number)
        self.state_for_h = state
        for i in range(100):
            self.current_player = self.player_number
            self.simulator = Simulator(state)
            cur_node = self.selection()
            self.expansion(cur_node)
            simulation_result = self.simulation()
            self.backpropagation(simulation_result, cur_node)
        self.current_player = self.player_number
        actions = self.possible_actions(state)
        relevant_children = [child for child in self.root.children if child.num_visits > 0 and child.action in actions]
        best_node = max(relevant_children, key=lambda x: x.get_empirical_mean())
        return best_node.action

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

    def h(self, action, state=None):
        ret_val = 0
        for act in action:
            if state is None:
                state = self.state_for_h
            treasures = state["treasures"]
            if act[0] == 'plunder':
                my_ship = act[1]
                enemy_ship = act[2]
                my_collected_reward = sum(
                    [treasure['reward'] for treasure in treasures.values() if treasure['location'] == my_ship])
                enemy_collected_reward = sum(
                    [treasure['reward'] for treasure in treasures.values() if treasure['location'] == enemy_ship])
                ret_val += enemy_collected_reward - my_collected_reward
            if act[0] == 'collect':
                ret_val += treasures[act[2]]['reward']
            if act[0] == 'deposit':
                ret_val += treasures[act[2]]['reward']
        return ret_val
