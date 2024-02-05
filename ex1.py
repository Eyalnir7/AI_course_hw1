import search
import random
import math
from itertools import product


ids = ["111111111", "111111111"]


class OnePieceProblem(search.Problem):
    """This class implements a medical problem according to problem description file"""

    def __init__(self, initial):
        """Don't forget to implement the goal test
        You should change the initial to your own representation.
        search.Problem.__init__(self, initial) creates the root node"""
        initial_state = OnePieceState(initial["pirate_ships"], initial["treasures"],
                                      [ship[0] for ship in initial["marine_ships"].values()])
        search.Problem.__init__(self, initial_state)
        self.map = initial["map"]
        self.pirate_ships = initial["pirate_ships"]
        self.treasures = initial["treasures"]
        self.marine_ships = initial["marine_ships"]

    def atomic_actions(self, ship, state):
        """
        Returns all the atomic actions that can be executed by a ship
        :param ship: should be a tuple (ship_name, (x, y))
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
        if self.check_collect_treasure(ship, state):
            actions.append(("collect_treasure", ship[0]))

        # deposit treasure action
        if self.check_collect_treasure(ship, state):
            actions.append(("deposit_treasure", ship[0]))

        return actions

    def check_collect_treasure(self, ship, state):
        """
        Checks if the ship can collect a treasure
        :param ship: should be a tuple (ship_name, (x, y))
        :param state: the current state (OnePieceState object)
        """
        if len([treasure for treasure in state.treasures if state.treasures[treasure] == ship[1]]) == 2:
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

    def check_deposit_treasure(self, ship, state):
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
            actions.append(self.atomic_actions((ship, state.pirate_ships[ship]), state))
        return list(product(*actions))

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""

    def goal_test(self, state):
        """ Given a state, checks if this is the goal state.
         Returns True if it is, False otherwise."""

    def h(self, node):
        """ This is the heuristic. It gets a node (not a state,
        state can be accessed via node.state)
        and returns a goal distance estimate"""
        return 0

    """Feel free to add your own functions
    (-2, -2, None) means there was a timeout"""


class OnePieceState:
    def __init__(self, pirate_ships, treasures, marine_ships):
        self.pirate_ships = pirate_ships
        self.treasures = treasures
        self.marine_ships = marine_ships

    def __eq__(self, other):
        return self.pirate_ships == other.pirate_ships and self.treasures == other.treasures and self.marine_ships == other.marine_ships

    def __hash__(self):
        return hash((self.pirate_ships, self.treasures, self.marine_ships))

    def __str__(self):
        return f"Pirate ships: {self.pirate_ships}, Treasures: {self.treasures}, Marine ships: {self.marine_ships}"

    def __repr__(self):
        return f"State: Pirate ships: {self.pirate_ships}, Treasures: {self.treasures}, Marine ships: {self.marine_ships}"



def create_onepiece_problem(game):
    return OnePieceProblem(game)

