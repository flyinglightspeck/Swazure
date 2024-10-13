import argparse

import numpy as np

from orchestrator import Orchestrator
# from planner import Planner
# from src.planner.planner import PlannerMilky
from swarm_locator import SwarmLocatorShortest, SwarmLocatorAveraging, SwarmLocatorMoveBlocking, SwarmLocatorMoveSource

np.random.seed(101)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--shape',
                            choices=['chess_408', 'palm_725', 'dragon_1147', 'kangaroo_972', 'skateboard_1372', 'racecar_3720'],
                            default='chess_408',
                            help="Name of the .xyz file in the src/assets directory.")
    arg_parser.add_argument('--alg',
                            choices=['swazure'],
                            default='swazure',
                            help="Name of the algorithm to run.")
    arg_parser.add_argument('--solution',
                            choices=['shortest', 'averaging', 'move_blocking', 'move_blocking+', 'move_source', 'move_source+'],
                            default='move_blocking+',
                            help="Name of the solution to run. shortest: use the shortest shortest path, averaging: average the first two shotrtest paths when available, move_blocking+: use the shortest path and move the blocking FLSs, move_source+: move the source to find a common sweet FLS.")
    arg_parser.add_argument('--scale', required=False, type=float, default=13.6,
                            help='Scale factor for point cloud coordinates.')
    arg_parser.add_argument('--radius-beta', required=False, type=float, default=.5,
                            help="Ratio of FLS radius to the minimum distance between FLSs.")
    arg_parser.add_argument('--radius', required=False, type=float, default=0,
                            help="Set radius explicitly (cm). If set to a non-zero value radius_beta will be ignored.")
    arg_parser.add_argument('--sweet-range-min', required=False, type=float, default=6,
                            help="Sweet range start. The maximum working range of the tracking device (cm).")
    arg_parser.add_argument('--sweet-range-max', required=False, type=float, default=8, help="Sweet range end (cm).")
    arg_parser.add_argument('--decaying-range-min', required=False, type=float, default=8,
                            help="Decaying range start (cm).")
    arg_parser.add_argument('--decaying-range-max', required=False, type=float, default=30,
                            help="Decaying range end. The maximum working range of the tracking device (cm).")
    arg_parser.add_argument('--accurate', action='store_true', required=False, default=False,
                            help="Use fully accurate measurements.")
    arg_parser.add_argument('--dead-reckoning-angle', required=False, type=float, default=0,
                            help="Dead reckoning angle (degree).")
    arg_parser.add_argument('--steps-threshold', required=False, type=int, default=2,
                            help="Maximum amount of steps the source FLS explores as a factor of its radius.")
    arg_parser.add_argument('--weighted', action='store_true', required=False, default=False,
                            help="If passed, use euclidean distance as the weight in the shortest path computation. Otherwize use shortest hops.")
    args = arg_parser.parse_args()

    if args.alg == 'swazure':
        if args.solution == 'shortest':
            orchestrator = Orchestrator(args, alg=SwarmLocatorShortest)
        elif args.solution == 'averaging':
            orchestrator = Orchestrator(args, alg=SwarmLocatorAveraging)
        elif 'move_blocking' in args.solution:
            orchestrator = Orchestrator(args, alg=SwarmLocatorMoveBlocking)
        elif 'move_source' in args.solution:
            orchestrator = Orchestrator(args, alg=SwarmLocatorMoveSource)

    orchestrator.initialize()
    orchestrator.start()
