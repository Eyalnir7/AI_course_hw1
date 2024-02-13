import ex1
import search
import time


def timeout_exec(func, args=(), kwargs={}, timeout_duration=10, default=None):
    """This function will spawn a thread and run the given function
    using the args, kwargs and return the given default value if the
    timeout_duration is exceeded.
    """
    import threading

    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = default

        def run(self):
            # try:
            self.result = func(*args, **kwargs)
            # except Exception as e:
            #    self.result = (-3, -3, e)

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.is_alive():
        return default
    else:
        return it.result


def check_problem(p, search_method, timeout):
    """ Constructs a problem using ex1.create_wumpus_problem,
    and solves it using the given search_method with the given timeout.
    Returns a tuple of (solution length, solution time, solution)"""

    """ (-2, -2, None) means there was a timeout
    (-3, -3, ERR) means there was some error ERR during search """

    t1 = time.time()
    s = timeout_exec(search_method, args=[p], timeout_duration=timeout)
    t2 = time.time()

    if isinstance(s, search.Node):
        solve = s
        solution = list(map(lambda n: n.action, solve.path()))[1:]
        return (len(solution), t2 - t1, solution)
    elif s is None:
        return (-2, -2, None)
    else:
        return s


def solve_problems(problems):
    solved = 0
    for problem in problems:
        # try:
        p = ex1.create_onepiece_problem(problem)
        # except Exception as e:
        #     print("Error creating problem: ", e)
        #     return None
        timeout = 60
        result = check_problem(
            p, (lambda p: search.astar_search(p, p.h)), timeout)
        print(f"A* ", result)
        if result[2] != None:
            if result[0] != -3:
                solved = solved + 1


def main():
    print(ex1.ids)
    """Here goes the input you want to check"""
    problems = [
        {
            "map": [
                ['S', 'S', 'I', 'S'],
                ['S', 'S', 'S', 'S'],
                ['B', 'S', 'S', 'S'],
                ['S', 'S', 'S', 'S']
            ],
            "pirate_ships": {"pirate_ship_1": (2, 0)},
            "treasures": {'treasure_1': (0, 2)},
            "marine_ships": {'marine_1': [(1, 1), (1, 2), (2, 2), (2, 1)]}
        },

        {
            "map": [
                ['S', 'S', 'I', 'S', 'S'],
                ['S', 'S', 'S', 'S', 'S'],
                ['B', 'S', 'S', 'S', 'S'],
                ['S', 'S', 'S', 'I', 'S'],
                ['S', 'S', 'S', 'S', 'S']
            ],
            "pirate_ships": {"pirate_ship_1": (2, 0)},
            "treasures": {'treasure_1': (0, 2), 'treasure_2': (3, 3)},
            "marine_ships": {
                'marine_1': [(1, 1), (1, 2), (2, 2), (2, 1), (3, 1)],
                'marine_2': [(3, 4), (2, 4), (1, 4), (1, 3), (2, 3)]
            }
        },
        {
            'map': [['S', 'S', 'S', 'S', 'I'],
                    ['S', 'I', 'S', 'S', 'S'],
                    ['S', 'S', 'S', 'S', 'S'],
                    ['B', 'S', 'S', 'I', 'S']],
            'pirate_ships': {'pirate_ship_1': (3, 0)},
            'treasures': {'treasure_1': (3, 3), 'treasure_2': (1, 1), 'treasure_3': (0, 4)},
            'marine_ships': {'marine_1': [(3, 2)], 'marine_2': [(0, 2), (0, 3)], 'marine_3': [(3, 4), (2, 4)]},
        },
        {
            'map': [['S', 'S', 'S', 'S', 'I'],
                    ['S', 'I', 'S', 'S', 'S'],
                    ['S', 'S', 'S', 'S', 'S'],
                    ['B', 'S', 'S', 'I', 'S']],
            'pirate_ships': {'pirate_ship_1': (3, 0)},
            'treasures': {'treasure_1': (1, 1), 'treasure_2': (3, 3)},
            'marine_ships': {'marine_1': [(2, 3), (2, 4)], 'marine_2': [(0, 1), (0, 0)], 'marine_3': [(0, 0), (0, 1)],
                             'marine_4': [(2, 4), (2, 3)]},
        },
        {
            'map': [['S', 'S', 'S', 'I', 'S', 'S', 'S'],
                    ['S', 'I', 'S', 'S', 'S', 'S', 'I'],
                    ['S', 'S', 'I', 'S', 'S', 'I', 'S'],
                    ['S', 'B', 'S', 'I', 'S', 'S', 'S'],
                    ['S', 'S', 'S', 'S', 'S', 'I', 'S'],
                    ['S', 'S', 'S', 'I', 'S', 'S', 'S']],
            'pirate_ships': {'pirate_ship_1': (3, 1)},
            'treasures': {'treasure_1': (4, 5), 'treasure_2': (1, 1), 'treasure_3': (1, 6)},
            'marine_ships': {'marine_1': [(4, 1), (4, 2)], 'marine_2': [(3, 5), (3, 4), (4, 4)],
                             'marine_3': [(1, 2), (1, 3), (1, 4)], 'marine_4': [(5, 6)]},
        },
    ]

    new_problems = [
        # {
        #     "map": [
        #         ['S', 'S', 'I', 'S', 'S', 'S'],
        #         ['S', 'S', 'S', 'S', 'I', 'S'],
        #         ['B', 'S', 'S', 'S', 'I', 'S'],
        #         ['S', 'S', 'S', 'S', 'I', 'I']
        #     ],
        #     "pirate_ships": {"pirate_ship_1": (2, 0), "pirate_ship_2": (2, 0)},
        #     "treasures": {'treasure_1': (0, 2), 'treasure_2': (3, 5)},
        #     "marine_ships": {'marine_1': [(1, 1), (1, 2), (2, 2), (2, 1)], 'marine_2': [(2, 1)]}
        # },
        {
            "map": [
                ['S', 'S', 'S', 'I', 'S', 'S', 'S'],
                ['S', 'I', 'S', 'S', 'S', 'S', 'I'],
                ['S', 'S', 'I', 'S', 'S', 'I', 'S'],
                ['S', 'B', 'S', 'I', 'S', 'S', 'S'],
                ['S', 'S', 'S', 'S', 'S', 'I', 'S'],
                ['S', 'S', 'S', 'I', 'S', 'S', 'S']
            ],
            "pirate_ships": {"pirate_ship_1": (3, 1), "pirate_ship_2": (3, 1), "pirate_ship_3": (3, 1)},
            "treasures": {'treasure_1': (4, 5), 'treasure_2': (1, 1), 'treasure_3': (1, 6), 'treasure_4': (1, 6)},
            "marine_ships": {'marine_1': [(4, 1), (4, 2)], 'marine_2': [(3, 5), (3, 4), (4, 4)],
                             'marine_3': [(1, 2), (1, 3), (1, 4)], 'marine_4': [(5, 6)]},
        },
        # {
        #     'map': [['S', 'S', 'S', 'S', 'I'],
        #             ['S', 'I', 'S', 'S', 'S'],
        #             ['S', 'S', 'S', 'S', 'S'],
        #             ['B', 'S', 'S', 'I', 'S']],
        #     'pirate_ships': {'pirate_ship_1': (3, 0)},
        #     'treasures': {'treasure_1': (3, 3), 'treasure_2': (1, 1), 'treasure_3': (0, 4), 'treasure_4': (0, 4),
        #                   'treasure_5': (0, 4), 'treasure_6': (0, 4)},
        #     'marine_ships': {}
        # },
        # {
        #     'map': [['S', 'S', 'S', 'S', 'I'],
        #             ['S', 'I', 'S', 'S', 'S'],
        #             ['S', 'S', 'S', 'S', 'S'],
        #             ['B', 'S', 'S', 'I', 'S']],
        #     'pirate_ships': {'pirate_ship_1': (3, 0), 'pirate_ship_2': (3, 0)},
        #     'treasures': {'treasure_1': (1, 1), 'treasure_2': (3, 3)},
        #     'marine_ships': {'marine_1': [(3, 2), (2, 2), (2, 3), (2, 4)]}
        # },
        # {
        #     'map': [['S', 'S', 'I', 'S'],
        #             ['S', 'B', 'S', 'S'],
        #             ['S', 'S', 'I', 'I'],
        #             ['S', 'S', 'I', 'I']],
        #     'pirate_ships': {'pirate_ship_1': (1, 1)},
        #     'treasures': {'treasure_1': (2, 2), 'treasure_2': (2, 2), 'treasure_3': (3, 3)},
        #     'marine_ships': {'marine_1': [(0, 0), (0, 1), (0, 0), (1, 0), (2, 0)]}
        # }

    ]

    solve_problems(
        new_problems)


if __name__ == '__main__':
    main()
