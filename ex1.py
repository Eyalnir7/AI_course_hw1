import time

import search
import math
from itertools import product, chain, combinations
import copy

import utils

ids = ["325161669", "330053893"]
NPC = 'pirate'


def create_all_vectors(k, s):
    """
    :param k: number of options for each coordinate
    :param s: the length of the vector
    :return: list of all possible vectors of length s, where each coordinate is in the range {0, ... ,k-1}
    """
    if s == 0:
        return [[]]
    all_vectors = []
    prev_all_vectors = create_all_vectors(k, s - 1)
    for i in range(k):
        for vector in prev_all_vectors:
            all_vectors.append([i] + vector)
    return all_vectors


def powerset(iterable):
    """powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))


class OnePieceState:
    def __init__(self, pirate_ships: dict[str: list[tuple[int, int], set]], marine_ships, treasures_in_base,
                 treasures_in_ships, marine_ships_backwards):
        self.pirate_ships = pirate_ships
        self.marine_ships = marine_ships
        self.treasures_in_base = treasures_in_base
        self.treasures_in_ships = treasures_in_ships
        self.marine_ships_backwards = marine_ships_backwards  # For each marine ship, the direction it's facing
        self.tuple = None

    def __eq__(self, other):
        return self.pirate_ships == other.pirate_ships and self.marine_ships == other.marine_ships and \
               self.treasures_in_base == other.treasures_in_base and self.treasures_in_ships == \
               other.treasures_in_ships and self.marine_ships_backwards == other.marine_ships_backwards

    def __hash__(self):
        return hash(self.tuple)

    def __str__(self):
        return f"Pirate ships: {self.pirate_ships}, Marine ships: {self.marine_ships}, " \
               f"treasures in base: {self.treasures_in_base}"

    def __repr__(self):
        return f"State: Pirate ships: {self.pirate_ships}, Marine ships: {self.marine_ships}, " \
               f"treasures in base: {self.treasures_in_base}"

    def __deepcopy__(self, memodict={}):
        return OnePieceState(copy.deepcopy(self.pirate_ships), copy.deepcopy(self.marine_ships),
                             copy.deepcopy(self.treasures_in_base), copy.deepcopy(self.treasures_in_ships),
                             copy.deepcopy(self.marine_ships_backwards))

    def to_tuple(self):
        pirate_ships = tuple((ship, tuple(values[0]), tuple(values[1])) for ship, values in
                             self.pirate_ships.items())
        marine_ships = tuple((ship, tuple(values)) for ship, values in self.marine_ships.items())
        treasures_in_base = tuple(self.treasures_in_base)
        treasures_in_ships = tuple(self.treasures_in_ships)
        marine_ships_backwards = tuple((ship, value) for ship, value in self.marine_ships_backwards.items())
        self.tuple = (pirate_ships, marine_ships, treasures_in_base, treasures_in_ships, marine_ships_backwards)


class OnePieceProblem(search.Problem):
    """This class implements a medical problem according to problem description file"""

    def __init__(self, initial):
        """Don't forget to implement the goal test
        You should change the initial to your own representation.
        search.Problem.__init__(self, initial) creates the root node"""
        initial["pirate_ships"] = {ship: [values, set()]
                                   for ship, values in initial["pirate_ships"].items()}
        initial_state = OnePieceState(initial["pirate_ships"],
                                      {key: value[0] for key, value in initial["marine_ships"].items()}, set(), set(),
                                      {key: False for key in initial["marine_ships"]})
        search.Problem.__init__(self, initial_state)
        self.map = initial["map"]
        self.pirate_ships = initial["pirate_ships"]
        self.index_to_pirate = {i: key for i, key in enumerate(self.pirate_ships)}
        self.treasures = initial["treasures"]
        self.index_to_treasure = {i: key for i, key in enumerate(self.treasures)}
        self.treasure_to_index = {key: i for i, key in self.index_to_treasure.items()}
        self.opposite_treasures = {}
        for key, value in self.treasures.items():
            i = 0
            while (value[0], value[1], i) in self.opposite_treasures:
                i += 1
            self.opposite_treasures[(value[0], value[1], i)] = key

        self.initial_marine_ships = initial["marine_ships"]
        self.marine_ships = copy.deepcopy(self.initial_marine_ships)

        for i in range(len(self.map)):
            for j in range(len(self.map[0])):
                if self.map[i][j] == "B":
                    self.base = (i, j)

        if len(self.pirate_ships) > 1:
            treasures_power_set = powerset(self.index_to_treasure.keys())
            shortest_path_to_treasures = {}
            for treasure_set in treasures_power_set:
                simple_ships = {NPC: self.base}
                treasures = {self.index_to_treasure[key]: self.treasures[self.index_to_treasure[key]]
                             for key in treasure_set}
                simple_initial = {
                    "map": self.map,
                    "pirate_ships": simple_ships,
                    "treasures": treasures,
                    "marine_ships": copy.deepcopy(self.initial_marine_ships)
                }
                simple_problem = OnePieceProblem2(simple_initial)
                simple_solution = search.astar_search(simple_problem)
                shortest_path_to_treasures[treasure_set] = simple_solution.path()
                #  print(list(map(lambda n: n.state, simple_solution.path())))

            shortest_distance = float('inf')
            shortest_path = []
            for partition in create_all_vectors(len(self.pirate_ships), len(self.treasures)):
                treasures_per_pirate = self.extract_treasures_sets_from_vector(partition)
                length_of_path = max([len(shortest_path_to_treasures[pirate]) for pirate in treasures_per_pirate])
                if length_of_path < shortest_distance:
                    shortest_distance = length_of_path
                    shortest_path = [shortest_path_to_treasures[pirate] for pirate in treasures_per_pirate]
                    shortest_path = [[n.state for n in path] for path in shortest_path]

            self.shortest_path = shortest_path
            print(shortest_distance)
            print(shortest_path)
            print(self.combine_paths_to_state(shortest_path, shortest_distance))

    def combine_paths_to_state(self, paths, shortest_distance):
        """
        :param shortest_distance: the length of the longest path in paths (which is the length of the best solution)
        :param paths: list of paths, where each path is a list of states
        :return: a state that is the combination of all the paths
        """
        final_path = []
        for i in range(shortest_distance):
            pirate_ships = {}
            treasures_in_base = set()
            treasures_in_ships = set()
            marine_ships = None
            marine_ships_backwards = None
            for j, path in enumerate(paths):
                if i < len(path):
                    pirate_ships[self.index_to_pirate[j]] = copy.deepcopy(path[i].pirate_ships[NPC])
                    treasures_in_base = treasures_in_base.union(path[i].treasures_in_base)
                    treasures_in_ships = treasures_in_ships.union(path[i].treasures_in_ships)
                    marine_ships = copy.deepcopy(path[i].marine_ships)
                    marine_ships_backwards = copy.deepcopy(path[i].marine_ships_backwards)
                else:
                    pirate_ships[self.index_to_pirate[j]] = copy.deepcopy(path[-1].pirate_ships[NPC])
                    treasures_in_base = copy.deepcopy(path[-1].treasures_in_base)
                    treasures_in_ships = copy.deepcopy(path[-1].treasures_in_ships)
            final_path.append(OnePieceState(pirate_ships, marine_ships, treasures_in_base, treasures_in_ships,
                                            marine_ships_backwards))

        return final_path


    def extract_treasures_sets_from_vector(self, vector):
        treasures_per_pirate = [[] for _ in range(len(self.pirate_ships))]
        for i in range(len(vector)):
            treasures_per_pirate[vector[i]].append(i)
        treasures_per_pirate = [tuple(treasures_per_p) for treasures_per_p in treasures_per_pirate]
        return treasures_per_pirate

    def atomic_actions(self, ship_name, state):
        """
        Returns all the atomic actions that can be executed by a ship
        :param ship_name: name of the ship
        :param state: the state of the game
        """
        ship = (ship_name, *state.pirate_ships[ship_name])
        possible = [("sail", ship[0], (ship[1][0] + 1, ship[1][1])), ("sail", ship[0], (ship[1][0] - 1, ship[1][1])),
                    ("sail", ship[0], (ship[1][0], ship[1][1] + 1)), ("sail", ship[0], (ship[1][0], ship[1][1] - 1))]

        actions = []
        # sails actions
        for action in possible:
            if self.check_sail(action):
                actions.append(action)

        # wait action
        actions.append(("wait", ship[0]))

        # collect treasure action
        treasure = self.check_collect_treasure(ship, state)
        if treasure:
            actions.append(("collect_treasure", ship[0], treasure))

        # deposit treasure action
        if self.check_deposit_treasure(ship):
            actions.append(("deposit_treasure", ship[0]))

        return actions

    def check_collect_treasure(self, ship, state):
        """
        Checks if the ship can collect a treasure
        :param ship: should be a tuple (ship_name, (x, y), set of treasures)
        :param state: the state of the game
        """
        if len(ship[2]) >= 2:
            return False
        if ship[1][0] + 1 < len(self.map) and (ship[1][0] + 1, ship[1][1], 0) in self.opposite_treasures:
            return self.get_treasure_from_island((ship[1][0] + 1, ship[1][1]), state)
        if ship[1][0] - 1 >= 0 and (ship[1][0] - 1, ship[1][1], 0) in self.opposite_treasures:
            return self.get_treasure_from_island((ship[1][0] - 1, ship[1][1]), state)
        if ship[1][1] + 1 < len(self.map[0]) and (ship[1][0], ship[1][1] + 1, 0) in self.opposite_treasures:
            return self.get_treasure_from_island((ship[1][0], ship[1][1] + 1), state)
        if ship[1][1] - 1 >= 0 and (ship[1][0], ship[1][1] - 1, 0) in self.opposite_treasures:
            return self.get_treasure_from_island((ship[1][0], ship[1][1] - 1), state)
        return False

    def get_treasure_from_island(self, island, state):
        """
        :param island: tuple of the location of the island - (x, y). We assume the island has treasures on it
        :param state: the state of the game
        :return: if there is a treasure that hasn't been collected on the island, return it. Otherwise, return
        one of the treasures on the island
        """
        collected = state.treasures_in_base.union(state.treasures_in_ships)
        i = 0
        while (island[0], island[1], i) in self.opposite_treasures:
            if self.opposite_treasures[(island[0], island[1], i)] not in collected:
                return self.opposite_treasures[(island[0], island[1], i)]
            i += 1
        return self.opposite_treasures[(island[0], island[1], 0)]

    def check_sail(self, action):
        """
        Checks if the sail action is valid
        :param action: tuple of ("sail", ship_name, (x, y))
        """
        if action[2][0] < 0 or action[2][1] < 0 or action[2][0] >= len(self.map) or action[2][1] >= len(self.map[0]):
            return False
        if self.map[action[2][0]][action[2][1]] == "I":
            return False
        return True

    def check_deposit_treasure(self, ship):
        """
        Checks if the ship can deposit a treasure
        :param ship: should be a tuple (ship_name, (x, y))
        """
        return self.map[ship[1][0]][ship[1][1]] == "B"

    def actions(self, state):
        """Returns all the actions that can be executed in the given
        state. The result should be a tuple (or other iterable) of actions
        as defined in the problem description file"""
        actions = []
        for ship in state.pirate_ships:
            actions.append(self.atomic_actions(ship, state))
        return list(product(*actions))

    def result(self, state: OnePieceState, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        new_state = copy.deepcopy(state)
        self.move_marine_ships(new_state)
        for a in action:
            if a[0] == "sail":
                new_state.pirate_ships[a[1]][0] = a[2]
            if a[0] == "collect_treasure":
                new_state.pirate_ships[a[1]][1].add(a[2])
                new_state.treasures_in_ships.add(a[2])
            if a[0] == "deposit_treasure":
                new_state.treasures_in_base = new_state.treasures_in_base.union(new_state.pirate_ships[a[1]][1])
                new_state.pirate_ships[a[1]][1] = set()
                for treasure in state.pirate_ships[a[1]][1]:  # for every treasure that was in the ship
                    new_state.treasures_in_ships.discard(treasure)

        marine_locations = new_state.marine_ships.values()
        for ship, value in new_state.pirate_ships.items():
            if len(value[1]) == 0:
                continue
            if value[0] in marine_locations:
                new_state.pirate_ships[ship][1] = set()
                for treasure in value[1]:
                    new_state.treasures_in_ships.discard(treasure)
        new_state.to_tuple()
        return new_state

    def move_marine_ships(self, state):
        for ship in self.marine_ships:
            if len(self.marine_ships[ship]) == 1:
                continue
            cur_index = self.marine_ships[ship].index(state.marine_ships[ship])
            if cur_index == 0 and state.marine_ships_backwards[ship]:
                state.marine_ships[ship] = self.marine_ships[ship][1]
                state.marine_ships_backwards[ship] = False
            elif cur_index == len(self.marine_ships[ship]) - 1 and not state.marine_ships_backwards[ship]:
                state.marine_ships[ship] = self.marine_ships[ship][-2]
                state.marine_ships_backwards[ship] = True
            elif state.marine_ships_backwards[ship]:
                state.marine_ships[ship] = self.marine_ships[ship][cur_index - 1]
            else:
                state.marine_ships[ship] = self.marine_ships[ship][cur_index + 1]

    def goal_test(self, state):
        """ Given a state, checks if this is the goal state.
         Returns True if it is, False otherwise."""
        return len(state.treasures_in_base) == len(self.treasures)

    def h(self, node):
        """ This is the heuristic. It gets a node (not a state,
        state can be accessed via [node.state]
        and returns a goal distance estimate"""
        if len(self.pirate_ships) <= 1:
            return self.h_2(node)

    def h_ben(self, node):
        state = node.state
        distances = []
        for ship in state.pirate_ships:
            if len(state.pirate_ships[ship][1]) < 2:
                min_treasure_distance = math.inf
                for x, y, i in self.opposite_treasures:
                    if self.opposite_treasures[(x, y, i)] not in state.treasures_in_base and \
                            self.opposite_treasures[(x, y, i)] not in state.treasures_in_ships:
                        if self.l1_distance(state.pirate_ships[ship][0], (x, y)) < min_treasure_distance:
                            min_treasure_distance = self.l1_distance(state.pirate_ships[ship][0], (x, y))
                if min_treasure_distance == math.inf:
                    min_treasure_distance = 0
                distances.append(self.l1_distance(state.pirate_ships[ship][0], self.base) + min_treasure_distance)
            else:
                distances.append(self.l1_distance(state.pirate_ships[ship][0], self.base))
        return sum(distances)

    def h_eyal(self, node):
        state = node.state
        collected = state.treasures_in_base.union(state.treasures_in_ships)
        uncollected = {treasure: loc for treasure, loc in self.treasures.items() if treasure not in collected}
        ship_distances = []
        for ship, value in state.pirate_ships.items():
            ship_distance = 0
            ship_location = value[0]
            ship_distance += self.closest_treasures(uncollected, ship_location,
                                                    min(2 - len(value[1]), len(uncollected)))
            if len(value[1]) > 0 and len(uncollected) == 0:
                ship_distance += self.l1_distance(ship_location, self.base)
            ship_location = self.base

            while len(uncollected) > 0:
                ship_distance += self.closest_treasures(uncollected, ship_location, min(2, len(uncollected)))

            ship_distances.append(ship_distance)

        return max(ship_distances) / (len(ship_distances))

    def closest_treasures(self, uncollected, location, num_treasures):
        """
        :param uncollected: set of uncollected treasures
        :param location: location
        :param num_treasures: number of treasures to find (-1 if don't even hold any treasures)
        :return: the length of the shortest path between the location and the closest num_treasures treasures including
        the distance to the base and the treasures themselves
        """
        closest_treasures = (None, None)
        if num_treasures == 0:
            return 0
        min_length = math.inf
        if num_treasures == 2:
            for treasure1, treasure_loc1 in uncollected.items():
                for treasure2, treasure_loc2 in uncollected.items():
                    if treasure1 != treasure2:
                        length = self.l1_distance(location, treasure_loc1) + self.l1_distance(treasure_loc1,
                                                                                              treasure_loc2)
                        if length < min_length:
                            min_length = length
                            closest_treasures = (treasure1, treasure2)
            min_length += self.l1_distance(uncollected[closest_treasures[1]], self.base)
        if num_treasures == 1:
            for treasure, treasure_loc in uncollected.items():
                length = self.l1_distance(location, treasure_loc)
                if length < min_length:
                    min_length = length
                    closest_treasures = (treasure, None)
            min_length += self.l1_distance(uncollected[closest_treasures[0]], self.base)

        if closest_treasures[0]:
            uncollected.pop(closest_treasures[0])
        if closest_treasures[1]:
            uncollected.pop(closest_treasures[1])

        return min_length

    def h_1(self, node):
        uncollected_treasures = len(self.treasures) - len(node.state.treasures_in_base)
        return uncollected_treasures / len(self.pirate_ships)

    def h_2(self, node):
        treasure_locations = {treasure: math.inf for treasure in self.treasures}

        # update the treasure locations of treasures that are held by a ship
        for ship in node.state.pirate_ships.values():
            # ship is a list [location, set of treasures]
            if len(ship[1]) != 0:  # if the ship has treasures
                ship_distance = self.get_ship_distance(ship[0]) + 1  # +1 for the step of depositing the treasure
                for treasure in ship[1]:
                    treasure_locations[treasure] = min(treasure_locations[treasure], ship_distance)

        # update the treasure locations of treasures that are not held by a ship
        for treasure in treasure_locations:
            if treasure_locations[treasure] == math.inf:
                if treasure not in node.state.treasures_in_base:
                    treasure_locations[treasure] = self.get_treasure_distance(self.treasures[treasure])
                else:
                    treasure_locations[treasure] = 0

        return sum(treasure_locations.values()) / len(self.pirate_ships)

    def get_ship_distance(self, location):
        """
        :param location: tuple of the location of the ship - (x,y)
        :return: the l1 distance to the base. If all near edges are islands, returns infinity
        """
        upper_island = location[0] - 1 >= 0 and self.map[location[0] - 1][location[1]] == "I"
        lower_island = location[0] + 1 < len(self.map) and self.map[location[0] + 1][location[1]] == "I"
        left_island = location[1] - 1 >= 0 and self.map[location[0]][location[1] - 1] == "I"
        right_island = location[1] + 1 < len(self.map[0]) and self.map[location[0]][location[1] + 1] == "I"
        if upper_island and lower_island and left_island and right_island and location != self.base:
            return math.inf
        return self.l1_distance(location, self.base)

    def get_treasure_distance(self, location):
        """
        :param location: the location of the island of a treasure that isn't taken by a ship or in the base
        :return: get the distance of a treasure from its island to the base
        """
        legal_indexes = self.legal_indexes(location)
        if len(legal_indexes) == 0:
            return math.inf
        return min([self.l1_distance(self.base, index) for index in legal_indexes])

    def legal_indexes(self, loc):
        legal = []
        if loc[0] + 1 < len(self.map) and self.map[loc[0] + 1][loc[1]] != "I":
            legal.append((loc[0] + 1, loc[1]))
        if loc[0] - 1 >= 0 and self.map[loc[0] - 1][loc[1]] != "I":
            legal.append((loc[0] - 1, loc[1]))
        if loc[1] + 1 < len(self.map[0]) and self.map[loc[0]][loc[1] + 1] != "I":
            legal.append((loc[0], loc[1] + 1))
        if loc[1] - 1 >= 0 and self.map[loc[0]][loc[1] - 1] != "I":
            legal.append((loc[0], loc[1] - 1))
        return legal

    @staticmethod
    def l1_distance(loc1, loc2):
        return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])


class OnePieceProblem2(search.Problem):
    """This class implements a medical problem according to problem description file"""

    def __init__(self, initial):
        """Don't forget to implement the goal test
        You should change the initial to your own representation.
        search.Problem.__init__(self, initial) creates the root node"""
        initial["pirate_ships"] = {ship: [values, set()]
                                   for ship, values in initial["pirate_ships"].items()}
        initial_state = OnePieceState(initial["pirate_ships"],
                                      {key: value[0] for key, value in initial["marine_ships"].items()}, set(), set(),
                                      {key: False for key in initial["marine_ships"]})
        search.Problem.__init__(self, initial_state)
        self.map = initial["map"]
        self.pirate_ships = initial["pirate_ships"]
        self.treasures = initial["treasures"]
        self.opposite_treasures = {}
        for key, value in self.treasures.items():
            i = 0
            while (value[0], value[1], i) in self.opposite_treasures:
                i += 1
            self.opposite_treasures[(value[0], value[1], i)] = key

        self.initial_marine_ships = initial["marine_ships"]
        self.marine_ships = copy.deepcopy(self.initial_marine_ships)

        for i in range(len(self.map)):
            for j in range(len(self.map[0])):
                if self.map[i][j] == "B":
                    self.base = (i, j)

    def atomic_actions(self, ship_name, state):
        """
        Returns all the atomic actions that can be executed by a ship
        :param ship_name: name of the ship
        :param state: the state of the game
        """
        ship = (ship_name, *state.pirate_ships[ship_name])
        possible = [("sail", ship[0], (ship[1][0] + 1, ship[1][1])), ("sail", ship[0], (ship[1][0] - 1, ship[1][1])),
                    ("sail", ship[0], (ship[1][0], ship[1][1] + 1)), ("sail", ship[0], (ship[1][0], ship[1][1] - 1))]

        actions = []
        # sails actions
        for action in possible:
            if self.check_sail(action):
                actions.append(action)

        # wait action
        actions.append(("wait", ship[0]))

        # collect treasure action
        treasure = self.check_collect_treasure(ship, state)
        if treasure:
            actions.append(("collect_treasure", ship[0], treasure))

        # deposit treasure action
        if self.check_deposit_treasure(ship):
            actions.append(("deposit_treasure", ship[0]))

        return actions

    def check_collect_treasure(self, ship, state):
        """
        Checks if the ship can collect a treasure
        :param ship: should be a tuple (ship_name, (x, y), set of treasures)
        :param state: the state of the game
        """
        if len(ship[2]) >= 2:
            return False
        if ship[1][0] + 1 < len(self.map) and (ship[1][0] + 1, ship[1][1], 0) in self.opposite_treasures:
            return self.get_treasure_from_island((ship[1][0] + 1, ship[1][1]), state)
        if ship[1][0] - 1 >= 0 and (ship[1][0] - 1, ship[1][1], 0) in self.opposite_treasures:
            return self.get_treasure_from_island((ship[1][0] - 1, ship[1][1]), state)
        if ship[1][1] + 1 < len(self.map[0]) and (ship[1][0], ship[1][1] + 1, 0) in self.opposite_treasures:
            return self.get_treasure_from_island((ship[1][0], ship[1][1] + 1), state)
        if ship[1][1] - 1 >= 0 and (ship[1][0], ship[1][1] - 1, 0) in self.opposite_treasures:
            return self.get_treasure_from_island((ship[1][0], ship[1][1] - 1), state)
        return False

    def get_treasure_from_island(self, island, state):
        """
        :param island: tuple of the location of the island - (x, y). We assume the island has treasures on it
        :param state: the state of the game
        :return: if there is a treasure that hasn't been collected on the island, return it. Otherwise, return
        one of the treasures on the island
        """
        collected = state.treasures_in_base.union(state.treasures_in_ships)
        i = 0
        while (island[0], island[1], i) in self.opposite_treasures:
            if self.opposite_treasures[(island[0], island[1], i)] not in collected:
                return self.opposite_treasures[(island[0], island[1], i)]
            i += 1
        return self.opposite_treasures[(island[0], island[1], 0)]

    def check_sail(self, action):
        """
        Checks if the sail action is valid
        :param action: tuple of ("sail", ship_name, (x, y))
        """
        if action[2][0] < 0 or action[2][1] < 0 or action[2][0] >= len(self.map) or action[2][1] >= len(self.map[0]):
            return False
        if self.map[action[2][0]][action[2][1]] == "I":
            return False
        return True

    def check_deposit_treasure(self, ship):
        """
        Checks if the ship can deposit a treasure
        :param ship: should be a tuple (ship_name, (x, y))
        """
        return self.map[ship[1][0]][ship[1][1]] == "B"

    def actions(self, state):
        """Returns all the actions that can be executed in the given
        state. The result should be a tuple (or other iterable) of actions
        as defined in the problem description file"""
        actions = []
        for ship in state.pirate_ships:
            actions.append(self.atomic_actions(ship, state))
        return list(product(*actions))

    def result(self, state: OnePieceState, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        new_state = copy.deepcopy(state)
        self.move_marine_ships(new_state)
        for a in action:
            if a[0] == "sail":
                new_state.pirate_ships[a[1]][0] = a[2]
            if a[0] == "collect_treasure":
                new_state.pirate_ships[a[1]][1].add(a[2])
                new_state.treasures_in_ships.add(a[2])
            if a[0] == "deposit_treasure":
                new_state.treasures_in_base = new_state.treasures_in_base.union(new_state.pirate_ships[a[1]][1])
                new_state.pirate_ships[a[1]][1] = set()
                for treasure in state.pirate_ships[a[1]][1]:  # for every treasure that was in the ship
                    new_state.treasures_in_ships.discard(treasure)

        marine_locations = new_state.marine_ships.values()
        for ship, value in new_state.pirate_ships.items():
            if len(value[1]) == 0:
                continue
            if value[0] in marine_locations:
                new_state.pirate_ships[ship][1] = set()
                for treasure in value[1]:
                    new_state.treasures_in_ships.discard(treasure)
        new_state.to_tuple()
        return new_state

    def move_marine_ships(self, state):
        for ship in self.marine_ships:
            if len(self.marine_ships[ship]) == 1:
                continue
            cur_index = self.marine_ships[ship].index(state.marine_ships[ship])
            if cur_index == 0 and state.marine_ships_backwards[ship]:
                state.marine_ships[ship] = self.marine_ships[ship][1]
                state.marine_ships_backwards[ship] = False
            elif cur_index == len(self.marine_ships[ship]) - 1 and not state.marine_ships_backwards[ship]:
                state.marine_ships[ship] = self.marine_ships[ship][-2]
                state.marine_ships_backwards[ship] = True
            elif state.marine_ships_backwards[ship]:
                state.marine_ships[ship] = self.marine_ships[ship][cur_index - 1]
            else:
                state.marine_ships[ship] = self.marine_ships[ship][cur_index + 1]

    def goal_test(self, state):
        """ Given a state, checks if this is the goal state.
         Returns True if it is, False otherwise."""
        return len(state.treasures_in_base) == len(self.treasures)

    def h(self, node):
        """ This is the heuristic. It gets a node (not a state,
        state can be accessed via [node.state]
        and returns a goal distance estimate"""
        # state = node.state
        # ship_avg_dist = {}
        # collected = state.treasures_in_base.union(state.treasures_in_ships)
        # uncollected = {treasure: loc for treasure, loc in self.treasures.items() if treasure not in collected}
        # uncollected_count = len(uncollected)
        #
        # # Fill the ship_avg_dist dictionary
        # for ship, value in state.pirate_ships.items():
        #     dist_to_base = self.get_ship_distance(value[0])
        #     if len(value[1]) == 1:
        #         avg = 0
        #         for treasure_loc in uncollected.values():
        #             avg += self.l1_distance(value[0], treasure_loc)
        #         avg = avg/uncollected_count if uncollected_count > 0 else 0
        #         ship_avg_dist[ship] = (dist_to_base, avg)
        #     else:
        #         ship_avg_dist[ship] = (dist_to_base, None)
        #
        # # Calculate the avg distance between every two uncollected treasures
        # avg_treasure_dist = 0
        # uncollected_vals = list(uncollected.values())
        # if uncollected_count > 1:
        #     for i in range(uncollected_count):
        #         for j in range(i + 1, uncollected_count):
        #             avg_treasure_dist += self.l1_distance(uncollected_vals[i], uncollected_vals[j])
        #
        #     avg_treasure_dist = avg_treasure_dist / (uncollected_count * (uncollected_count - 1) / 2)
        #
        # # Calculate the avg uncollected treasure distance to the base
        # avg_treasure_to_base = 0
        # for treasure_loc in uncollected_vals:
        #     avg_treasure_to_base += self.get_treasure_distance(treasure_loc)
        # avg_treasure_to_base = avg_treasure_to_base / uncollected_count if uncollected_count > 0 else 0
        #
        # avg_uncollected_per_ship = uncollected_count / len(self.pirate_ships)
        #
        # heuristic_per_ship = []
        # for ship, values in state.pirate_ships.items():
        #     if len(values[1]) == 0:
        #         heuristic_per_ship.append(avg_uncollected_per_ship/2 * avg_treasure_dist
        #                                  + avg_uncollected_per_ship * avg_treasure_to_base)
        #     if len(values[1]) == 1:
        #         heuristic_per_ship.append(ship_avg_dist[ship][0] + ship_avg_dist[ship][1]
        #                                  + (avg_uncollected_per_ship-1)/2 * avg_treasure_dist
        #                                  + (avg_uncollected_per_ship-1) * avg_treasure_to_base)
        #     if len(values[1]) > 1:
        #         heuristic_per_ship.append(ship_avg_dist[ship][0] + avg_uncollected_per_ship/2 * avg_treasure_dist
        #                                  + avg_uncollected_per_ship * avg_treasure_to_base)
        return self.h_eyal(node)

    def h_ben(self, node):
        state = node.state
        distances = []
        for ship in state.pirate_ships:
            if len(state.pirate_ships[ship][1]) < 2:
                min_treasure_distance = math.inf
                for x, y, i in self.opposite_treasures:
                    if self.opposite_treasures[(x, y, i)] not in state.treasures_in_base and \
                            self.opposite_treasures[(x, y, i)] not in state.treasures_in_ships:
                        if self.l1_distance(state.pirate_ships[ship][0], (x, y)) < min_treasure_distance:
                            min_treasure_distance = self.l1_distance(state.pirate_ships[ship][0], (x, y))
                if min_treasure_distance == math.inf:
                    min_treasure_distance = 0
                distances.append(self.l1_distance(state.pirate_ships[ship][0], self.base) + min_treasure_distance)
            else:
                distances.append(self.l1_distance(state.pirate_ships[ship][0], self.base))
        return sum(distances)

    def h_eyal(self, node):
        state = node.state
        collected = state.treasures_in_base.union(state.treasures_in_ships)
        uncollected = {treasure: loc for treasure, loc in self.treasures.items() if treasure not in collected}
        ship_distances = []
        for ship, value in state.pirate_ships.items():
            ship_distance = 0
            ship_location = value[0]
            ship_distance += self.closest_treasures(uncollected, ship_location,
                                                    min(2 - len(value[1]), len(uncollected)))
            if len(value[1]) > 0 and len(uncollected) == 0:
                ship_distance += self.l1_distance(ship_location, self.base)
            ship_location = self.base

            while len(uncollected) > 0:
                ship_distance += self.closest_treasures(uncollected, ship_location, min(2, len(uncollected)))

            ship_distances.append(ship_distance)

        return max(ship_distances) / (len(ship_distances))

    def closest_treasures(self, uncollected, location, num_treasures):
        """
        :param uncollected: set of uncollected treasures
        :param location: location
        :param num_treasures: number of treasures to find (-1 if don't even hold any treasures)
        :return: the length of the shortest path between the location and the closest num_treasures treasures including
        the distance to the base and the treasures themselves
        """
        closest_treasures = (None, None)
        if num_treasures == 0:
            return 0
        min_length = math.inf
        if num_treasures == 2:
            for treasure1, treasure_loc1 in uncollected.items():
                for treasure2, treasure_loc2 in uncollected.items():
                    if treasure1 != treasure2:
                        length = self.l1_distance(location, treasure_loc1) + self.l1_distance(treasure_loc1,
                                                                                              treasure_loc2)
                        if length < min_length:
                            min_length = length
                            closest_treasures = (treasure1, treasure2)
            min_length += self.l1_distance(uncollected[closest_treasures[1]], self.base)
        if num_treasures == 1:
            for treasure, treasure_loc in uncollected.items():
                length = self.l1_distance(location, treasure_loc)
                if length < min_length:
                    min_length = length
                    closest_treasures = (treasure, None)
            min_length += self.l1_distance(uncollected[closest_treasures[0]], self.base)

        if closest_treasures[0]:
            uncollected.pop(closest_treasures[0])
        if closest_treasures[1]:
            uncollected.pop(closest_treasures[1])

        return min_length

    def h_1(self, node):
        uncollected_treasures = len(self.treasures) - len(node.state.treasures_in_base)
        return uncollected_treasures / len(self.pirate_ships)

    def h_2(self, node):
        treasure_locations = {treasure: math.inf for treasure in self.treasures}

        # update the treasure locations of treasures that are held by a ship
        for ship in node.state.pirate_ships.values():
            # ship is a list [location, set of treasures]
            if len(ship[1]) != 0:  # if the ship has treasures
                ship_distance = self.get_ship_distance(ship[0]) + 1  # +1 for the step of depositing the treasure
                for treasure in ship[1]:
                    treasure_locations[treasure] = min(treasure_locations[treasure], ship_distance)

        # update the treasure locations of treasures that are not held by a ship
        for treasure in treasure_locations:
            if treasure_locations[treasure] == math.inf:
                if treasure not in node.state.treasures_in_base:
                    treasure_locations[treasure] = self.get_treasure_distance(self.treasures[treasure])
                else:
                    treasure_locations[treasure] = 0

        return sum(treasure_locations.values()) / len(self.pirate_ships)

    def get_ship_distance(self, location):
        """
        :param location: tuple of the location of the ship - (x,y)
        :return: the l1 distance to the base. If all near edges are islands, returns infinity
        """
        upper_island = location[0] - 1 >= 0 and self.map[location[0] - 1][location[1]] == "I"
        lower_island = location[0] + 1 < len(self.map) and self.map[location[0] + 1][location[1]] == "I"
        left_island = location[1] - 1 >= 0 and self.map[location[0]][location[1] - 1] == "I"
        right_island = location[1] + 1 < len(self.map[0]) and self.map[location[0]][location[1] + 1] == "I"
        if upper_island and lower_island and left_island and right_island and location != self.base:
            return math.inf
        return self.l1_distance(location, self.base)

    def get_treasure_distance(self, location):
        """
        :param location: the location of the island of a treasure that isn't taken by a ship or in the base
        :return: get the distance of a treasure from its island to the base
        """
        legal_indexes = self.legal_indexes(location)
        if len(legal_indexes) == 0:
            return math.inf
        return min([self.l1_distance(self.base, index) for index in legal_indexes])

    def legal_indexes(self, loc):
        legal = []
        if loc[0] + 1 < len(self.map) and self.map[loc[0] + 1][loc[1]] != "I":
            legal.append((loc[0] + 1, loc[1]))
        if loc[0] - 1 >= 0 and self.map[loc[0] - 1][loc[1]] != "I":
            legal.append((loc[0] - 1, loc[1]))
        if loc[1] + 1 < len(self.map[0]) and self.map[loc[0]][loc[1] + 1] != "I":
            legal.append((loc[0], loc[1] + 1))
        if loc[1] - 1 >= 0 and self.map[loc[0]][loc[1] - 1] != "I":
            legal.append((loc[0], loc[1] - 1))
        return legal

    @staticmethod
    def l1_distance(loc1, loc2):
        return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])


"""Feel free to add your own functions
(-2, -2, None) means there was a timeout"""


def create_onepiece_problem(game):
    return OnePieceProblem(game)
