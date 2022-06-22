import argparse
import collections
import math
import time

import numpy as np
import pyffish as sf
from tqdm import tqdm

import uci


class UCTNode():
    def __init__(self, game_state, move, parent=None):
        self.game_state = game_state
        self.move = move
        self.is_expanded = False
        self.parent = parent
        self.children = {}
        self.num_moves = len(self.game_state.legal_moves)
        self.child_priors = np.zeros([self.num_moves], dtype=np.float32)
        self.child_total_value = np.zeros([self.num_moves], dtype=np.float32)
        self.child_number_visits = np.zeros([self.num_moves], dtype=np.float32)

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
        return 0.2 * np.sqrt(math.log(self.number_visits + 1) / (1 + self.child_number_visits))

    def best_child(self):
        return np.argmax(self.child_Q() + self.child_U())

    def select_leaf(self):
        current = self
        while current.is_expanded:
            best_move = current.best_child()
            current = current.maybe_add_child(best_move)
        return current

    def expand(self, child_priors, initial_values):
        self.is_expanded = True
        self.child_priors = child_priors
        self.child_total_value = initial_values

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

    def best_move(self):
        # pick best score in case of equal visit count
        return np.argmax(self.child_number_visits + self.child_Q())

    def pv(self):
        current = self
        while current.is_expanded:
            best_move = current.best_move()
            if best_move in current.children:
                current = current.children[best_move]
            else:
                break
        return current.game_state.get_san_moves()

    def traverse(self, apply=lambda x: True):
        if apply(self):
            if not self.children:
                return
            for child in sorted(self.children.values(), key=lambda x: x.number_visits, reverse=True):
                child.traverse(apply)

    def __repr__(self):
        fen = self.game_state.get_fen()
        sans = [sf.get_san(self.game_state.variant, fen, m) for m in self.game_state.legal_moves]
        moves = sorted(zip(sans, self.child_Q(), self.child_number_visits),
                       key=lambda x: x[2] + x[1], reverse=True)
        return 'Position: {}\nMoves: {}'.format(self.game_state.get_fen(),
                ', '.join('{}: {:.4f} ({:.0f})'.format(*i) for i in moves if i[2]))


class PreRootNode(object):
    def __init__(self):
        self.parent = None
        self.child_total_value = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)

    @property
    def number_visits(self):
        return sum(self.child_number_visits.values())


class EnginePolicy():
    def __init__(self, path, options, limits):
        self.engine = uci.Engine([path], options)
        self.engine.setoption('UCI_Variant', args.variant)
        self.engine.newgame()
        self.limits = limits

    def evaluate(self, game_state):
        num_moves = len(game_state.legal_moves)
        if num_moves:
            self.engine.position(game_state.fen, game_state.move_stack)
            multipv = self.engine.go(**self.limits)
            max_score = multipv[1]['score']
            min_score = min(s['score'] for s in multipv.values())
            priors = np.full([num_moves], multipv[1]['depth'], dtype=np.float32)
            values = np.full([num_moves], max(min_score - 2 * (max_score - min_score + 0.05), -1), dtype=np.float32)
            for info in multipv.values():
                values[game_state.legal_moves.index(info['pv'][0])] = info['score']
            return priors, values, max_score * game_state.side_to_move
        else:
            result = sf.game_result(game_state.variant, game_state.fen, game_state.move_stack)
            return None, None, (1 if result > 0 else -1 if result < 0 else 0) * game_state.side_to_move


class RandomPolicy():
    @staticmethod
    def evaluate(game_state):
        num_moves = len(game_state.legal_moves)
        if num_moves:
            return np.zeros([num_moves], dtype=np.float32), np.zeros([num_moves], dtype=np.float32), np.random.triangular(-1, 0, 1)
        else:
            result = sf.game_result(game_state.variant, game_state.fen, game_state.move_stack)
            return None, None, (1 if result > 0 else -1 if result < 0 else 0) * game_state.side_to_move


def uct_search(game_state, num_reads, policy):
    root = UCTNode(game_state, move=None, parent=PreRootNode())
    for _ in tqdm(range(num_reads)):
        leaf = root.select_leaf()
        child_priors, initial_values, value_estimate = policy.evaluate(leaf.game_state)
        if child_priors is not None:
            leaf.expand(child_priors, initial_values)
        leaf.backup(value_estimate)
    return root


class GameState():
    def __init__(self, variant="chess", fen=None, moves=None):
        self.variant = variant
        self.fen = fen or sf.start_fen(variant)
        self.move_stack = moves or []
        self.side_to_move = 1 if (self.fen.split(" ")[1] == "w") == (len(self.move_stack) % 2 == 0) else -1
        self.legal_moves = sf.legal_moves(self.variant, self.fen, self.move_stack)

    def play(self, move):
        new_move_stack = self.move_stack + [self.legal_moves[move]]
        return GameState(self.variant, self.fen, new_move_stack)

    def get_fen(self):
        return sf.get_fen(self.variant, self.fen, self.move_stack)

    def get_san_moves(self):
        return sf.get_san_moves(self.variant, self.fen, self.move_stack)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--engine', help='chess variant engine path, e.g., to Fairy-Stockfish')
    parser.add_argument('-o', '--ucioptions', type=lambda kv: kv.split("="), action='append', default=[],
                        help='UCI option as key=value pair. Repeat to add more options.')
    parser.add_argument('-v', '--variant', default='chess', help='variant to analyze')
    parser.add_argument('-f', '--fen', help='FEN to analyze')
    parser.add_argument('-m', '--moves', default='', help='moves from FEN')
    parser.add_argument('-r', '--rollouts', type=int, default=100, help='number of rollouts')
    parser.add_argument('-d', '--depth', type=int, default=None, help='engine search depth')
    parser.add_argument('-t', '--movetime', type=int, default=None, help='engine search movetime (ms)')
    parser.add_argument('-p', '--print-tree', action='store_true', help='print search tree')
    parser.add_argument('-b', '--export-book', help='export as EPD book')
    parser.add_argument('--min-visits', type=int, default=5, help='only print/export nodes with minimum visit count')
    parser.add_argument('--min-ratio', type=float, default=0.03, help='only print/export nodes with minimum visit ratio')
    args = parser.parse_args()

    # Init engine
    limits = dict()
    if args.depth:
        limits['depth'] = args.depth
    if args.movetime:
        limits['movetime'] = args.movetime
    if not limits:
        limits['movetime'] = int(math.sqrt(args.rollouts))
    options = dict(args.ucioptions)
    sf.set_option('VariantPath', options.get('VariantPath', ''))
    if args.variant not in sf.variants():
        raise Exception('Variant {} not supported'.format(args.variant))
    options.setdefault('multipv', '3')
    if args.engine:
        policy = EnginePolicy(args.engine, options, limits)
    else:
        policy = RandomPolicy()

    # UCT search
    root_pos = GameState(args.variant, args.fen, args.moves.split(' ') if args.moves else None)
    start = time.perf_counter()
    root_node = uct_search(root_pos, args.rollouts, policy)
    end = time.perf_counter()
    print('Runtime: {:.3f} s'.format(end - start))
    try:
        import resource
        print('Memory: {} KB'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    except ImportError:
        pass
    print('PV: {}'.format(' '.join(root_node.pv())))
    print(root_node)
    if args.print_tree:
        def print_node(node):
            if node.number_visits >= args.min_visits and node.number_visits / node.parent.number_visits >= args.min_ratio:
                print('{}: {:.0f} ({:.3f})'.format(' '.join(node.game_state.get_san_moves()), node.number_visits,
                                                    node.total_value / node.number_visits))
                return True
            return False
        print('\nTree')
        root_node.traverse(apply=print_node)
    if args.export_book:
        with open(args.export_book, 'w') as epd:
            def write_epd(node):
                if node.number_visits >= args.min_visits and node.number_visits / node.parent.number_visits >= args.min_ratio:
                    epd.write(node.game_state.get_fen() + '\n')
                    return True
                return False
            root_node.traverse(apply=write_epd)
