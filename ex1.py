import search
import random
import math
from itertools import product
import copy


ids = ["111111111", "111111111"]


class OnePieceState:
    def __init__(self, pirate_ships, marine_ships, treasures_in_base):
        self.pirate_ships = pirate_ships
        self.marine_ships = marine_ships
        self.treasures_in_base = treasures_in_base

    def __eq__(self, other):
        return self.pirate_ships == other.pirate_ships and self.marine_ships == other.marine_ships

    def __hash__(self):
        return hash((self.pirate_ships, self.marine_ships, self.treasures_in_base))

    def __str__(self):
        return f"Pirate ships: {self.pirate_ships}, Marine ships: {self.marine_ships}, " \
               f"treasures in base: {self.treasures_in_base}"

    def __repr__(self):
        return f"State: Pirate ships: {self.pirate_ships}, Marine ships: {self.marine_ships}, " \
               f"treasures in base: {self.treasures_in_base}"

    def __deepcopy__(self, memodict={}):
        return OnePieceState(copy.deepcopy(self.pirate_ships), copy.deepcopy(self.marine_ships),
                             copy.deepcopy(self.treasures_in_base))

    def to_tuple(self):
        pirate_ships = tuple((ship, tuple(values[0]), tuple(values[1])) for ship, values in self.pirate_ships.items())


class OnePieceProblem(search.Problem):
    """This class implements a medical problem according to problem description file"""

    def __init__(self, initial):
        """Don't forget to implement the goal test
        You should change the initial to your own representation.
        search.Problem.__init__(self, initial) creates the root node"""
        initial["pirate_ships"] = {ship: [values, set()]
                                   for ship, values in initial["pirate_ships"].items()}
        initial_state = OnePieceState(initial["pirate_ships"],
                                      {key: value[0] for key, value in initial["marine_ships"].items()}, set())
        search.Problem.__init__(self, initial_state)
        self.map = initial["map"]
        self.pirate_ships = initial["pirate_ships"]
        self.treasures = initial["treasures"]
        self.initial_marine_ships = initial["marine_ships"]
        self.marine_ships = copy.deepcopy(self.initial_marine_ships)

        for i in range(len(self.map)):
            for j in range(len(self.map[0])):
                if self.map[i][j] == "B":
                    self.base = (i, j)

    def atomic_actions(self, ship):
        """
        Returns all the atomic actions that can be executed by a ship
        :param ship: should be a tuple (ship_name, (x, y), set of treasures)
        :param state: should be a OnePieceState object
        """
        actions = [("sail", ship[0], (ship[1][0] + 1, ship[1][1])), ("sail", ship[0], (ship[1][0] - 1, ship[1][1])),
                   ("sail", ship[0], (ship[1][0], ship[1][1] + 1)), ("sail", ship[0], (ship[1][0], ship[1][1] - 1))]
        # sails actions
        for action in actions:
            if not self.check_sail(action):
                actions.remove(action)

        # wait action
        actions.append(("wait", ship[0]))

        # collect treasure action
        if self.check_collect_treasure(ship):
            actions.append(("collect_treasure", ship[0]))

        # deposit treasure action
        if self.check_deposit_treasure(ship):
            actions.append(("deposit_treasure", ship[0]))

        return actions

    def check_collect_treasure(self, ship):
        """
        Checks if the ship can collect a treasure
        :param ship: should be a tuple (ship_name, (x, y), set of treasures)
        :param state: the current state (OnePieceState object)
        """
        if len(ship[2]) == 2:
            return False
        if ship[1][0]+1 < len(self.map) and self.map[ship[1][0]+1][ship[1][1]] == "I":
            return True
        if ship[1][0]-1 >= 0 and self.map[ship[1][0]-1][ship[1][1]] == "I":
            return True
        if ship[1][1]+1 < len(self.map[0]) and self.map[ship[1][0]][ship[1][1]+1] == "I":
            return True
        if ship[1][1]-1 >= 0 and self.map[ship[1][0]][ship[1][1]-1] == "I":
            return True
        return False

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
        :param state: the current state (OnePieceState object)
        """
        return self.map[ship[1][0]][ship[1][1]] == "B"

    def actions(self, state):
        """Returns all the actions that can be executed in the given
        state. The result should be a tuple (or other iterable) of actions
        as defined in the problem description file"""
        actions = []
        for ship in state.pirate_ships:
            actions.append(self.atomic_actions((ship, *state.pirate_ships[ship])))
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
            if a[0] == "deposit_treasure":
                new_state.treasures_in_base = new_state.treasures_in_base.union(new_state.pirate_ships[a[1]][1])
                new_state.pirate_ships[a[1]][1] = set()

        marine_locations = new_state.marine_ships.values()
        for ship, value in new_state.pirate_ships.items():
            if value[0] in marine_locations:
                new_state.pirate_ships[ship][1] = set()

    def move_marine_ships(self, state):
        for ship in self.marine_ships:
            cur_index = self.marine_ships[ship].index(state.marine_ships[ship])
            if cur_index == len(self.marine_ships[ship])-1:
                cur_index = 0
                self.marine_ships[ship] = self.marine_ships[ship][::-1]
            state.marine_ships[ship] = self.marine_ships[ship][self.marine_ships[cur_index]+1]

    def goal_test(self, state):
        """ Given a state, checks if this is the goal state.
         Returns True if it is, False otherwise."""
        return len(state.treasures_in_base) == len(self.treasures)

    def h(self, node):
        """ This is the heuristic. It gets a node (not a state,
        state can be accessed via node.state)
        and returns a goal distance estimate"""
        return 0

    def h_1(self, node):
        uncollected_treasures = len(self.treasures) - len(node.state.treasures_in_base)
        return uncollected_treasures / len(self.pirate_ships)

    def h_2(self, node):
        treasure_locations = {treasure: self.get_treasure_distance(loc) for treasure, loc in self.treasures.items()}

        for ship in node.state.pirate_ships.values():
            if len(ship[1]) == 0:
                continue
            ship_distance = self.get_treasure_distance(ship[0])
            for treasure in ship[1]:
                treasure_locations[treasure] = min(treasure_locations[treasure], ship_distance)

        return sum(treasure_locations.values())/len(self.pirate_ships)

    def get_treasure_distance(self, location):
        legal_indexes = self.legal_indexes(location)
        if len(legal_indexes) == 0:
            return math.inf
        return min([self.l1_distance(self.base, index) for index in legal_indexes])

    def legal_indexes(self, loc):
        indexes = []
        if loc[0] + 1 < len(self.map) and self.map[loc[0] + 1][loc[1]] != "I":
            indexes.append((loc[0] + 1, loc[1]))
        if loc[0] - 1 >= 0 and self.map[loc[0 - 1]][loc[1]] != "I":
            indexes.append((loc[0] - 1, loc[1]))
        if loc[1] + 1 < len(self.map[0]) and self.map[loc[0]][loc[1] + 1] != "I":
            indexes.append((loc[0], loc[1] + 1))
        if loc[1] - 1 >= 0 and self.map[loc[0]][loc[1 - 1]] != "I":
            indexes.append((loc[0], loc[1] - 1))
        return indexes

    @staticmethod
    def l1_distance(loc1, loc2):
        return abs(loc1[0]-loc2[0]) + abs(loc1[1]-loc2[1])


"""Feel free to add your own functions
(-2, -2, None) means there was a timeout"""


def create_onepiece_problem(game):
    return OnePieceProblem(game)
