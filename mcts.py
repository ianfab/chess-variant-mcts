import argparse
import collections
import math
import resource

import numpy as np
import pyffish as sf

import uci


class UCTNode():
    def __init__(self, game_state, move, parent=None):
        self.game_state = game_state
        self.move = move
        self.is_expanded = False
        self.parent = parent
        self.children = {}
        num_moves = len(self.game_state.legal_moves)
        self.child_priors = np.zeros([num_moves], dtype=np.float32)
        self.child_total_value = np.zeros([num_moves], dtype=np.float32)
        self.child_number_visits = np.zeros([num_moves], dtype=np.float32)

    @property
    def number_visits(self):
        return self.parent.child_number_visits[self.move]

    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits[self.move] = value

    @property
    def total_value(self):
        return self.parent.child_total_value[self.move]

    @total_value.setter
    def total_value(self, value):
        self.parent.child_total_value[self.move] = value

    def child_Q(self):
        return self.child_total_value / (1 + self.child_number_visits)

    def child_U(self):
        return math.sqrt(self.number_visits) * (
                self.child_priors / (1 + self.child_number_visits))

    def best_child(self):
        return np.argmax(self.child_Q() + self.child_U())

    def select_leaf(self):
        current = self
        while current.is_expanded:
            best_move = current.best_child()
            current = current.maybe_add_child(best_move)
        return current

    def expand(self, child_priors):
        self.is_expanded = True
        self.child_priors = child_priors

    def maybe_add_child(self, move):
        if move not in self.children:
            self.children[move] = UCTNode(self.game_state.play(move), move, parent=self)
        return self.children[move]

    def backup(self, value_estimate: float):
        current = self
        while current.parent is not None:
            current.number_visits += 1
            current.total_value += value_estimate * -current.game_state.side_to_move
            current = current.parent


class PreRootNode(object):
    def __init__(self):
        self.parent = None
        self.child_total_value = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)


def uct_search(game_state, num_reads):
    root = UCTNode(game_state, move=None, parent=PreRootNode())
    for _ in range(num_reads):
        leaf = root.select_leaf()
        child_priors, value_estimate = Engine.evaluate(leaf.game_state)
        if child_priors is not None:
            leaf.expand(child_priors)
        leaf.backup(value_estimate)
    return root, np.argmax(root.child_number_visits)


class Engine():
    @classmethod
    def evaluate(self, game_state):
        num_moves = len(game_state.legal_moves)
        if num_moves:
            priors = np.ones([num_moves], dtype=np.float32) / num_moves
            engine.position(game_state.fen, game_state.move_stack)
            bestmove, evaluation = engine.go(**limits)
            priors[game_state.legal_moves.index(bestmove)] *= 2
            return priors, evaluation * game_state.side_to_move
        else:
            priors = None
            result = sf.game_result(game_state.variant, game_state.fen, game_state.move_stack)
            return priors, (1 if result > 0 else -1 if result < 0 else 0) * game_state.side_to_move


class GameState():
    def __init__(self, variant="chess", fen=None, moves=None):
        self.variant = variant
        self.fen = fen or sf.start_fen(variant)
        self.move_stack = moves or []
        self.side_to_move = 1 if (self.fen.split(" ")[1] == "w") == (len(self.move_stack) % 2 == 0) else -1
        self.legal_moves = sf.legal_moves(self.variant, self.fen, self.move_stack)

    def play(self, move):
        moves = self.move_stack + [self.legal_moves[move]]
        return GameState(self.variant, self.fen, moves)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--engine', required=True, help='chess variant engine path, e.g., to Fairy-Stockfish')
    parser.add_argument('-o', '--ucioptions', type=lambda kv: kv.split("="), action='append', default=[],
                        help='UCI option as key=value pair. Repeat to add more options.')
    parser.add_argument('-v', '--variant', default='chess', help='variant to run search on')
    parser.add_argument('-r', '--rollouts', type=int, default=100, help='number of rollouts')
    parser.add_argument('-d', '--depth', type=int, default=None, help='engine search depth')
    parser.add_argument('-t', '--movetime', type=int, default=None, help='engine search movetime (ms)')
    args = parser.parse_args()


engine = uci.Engine([args.engine], dict(args.ucioptions))
sf.set_option("VariantPath", engine.options.get("VariantPath", ""))
limits = dict()
if args.depth:
    limits['depth'] = args.depth
if args.movetime:
    limits['movetime'] = args.movetime
if not limits:
    parser.error('At least one of --depth and --movetime is required.')
engine.setoption('UCI_Variant', args.variant)
engine.newgame()


import time
tick = time.time()
root = GameState()
root_node, bestmove = uct_search(root, args.rollouts)
print(root_node.game_state.legal_moves[bestmove])
print(dict(zip(root_node.game_state.legal_moves, root_node.child_Q())))
print(dict(zip(root_node.game_state.legal_moves, root_node.child_number_visits)))
current = root_node
while current.is_expanded:
    best_move = np.argmax(current.child_number_visits)
    print(current.game_state.legal_moves[best_move])
    print(current.child_number_visits[best_move])
    print(dict(zip(current.game_state.legal_moves, current.child_number_visits)))
    if best_move in current.children:
        current = current.children[best_move]
    else:
        break

tock = time.time()
print('Runtime: {}'.format(tock - tick))
print('Memory: {} KB'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
