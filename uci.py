import subprocess
import threading
from collections.abc import Iterable
import math


class Engine():
    KEYWORDS = {'depth': int, 'seldepth': int, 'multipv': int, 'nodes': int,
                'nps': int, 'time': int, 'score': list, 'pv': list}

    def __init__(self, args, options=None):
        self.process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)
        self.lock = threading.Lock()
        self.options = options or {}
        self._init()

    def write(self, message):
        with self.lock:
            self.process.stdin.write(message)
            self.process.stdin.flush()

    def setoption(self, name, value):
        self.write('setoption name {} value {}\n'.format(name, value))

    def _init(self):
        self.write('uci\n')
        self.read('uciok')
        for option, value in self.options.items():
            self.setoption(option, value)

    def newgame(self):
        self.write('ucinewgame\n')
        self.write('isready\n')
        self.read('readyok')

    def position(self, fen=None, moves=None):
        sfen = 'fen {}'.format(fen) if fen else 'startpos'
        moves = 'moves {}'.format(' '.join(moves)) if moves else ''
        self.write('position {} {}\n'.format(sfen, moves))

    @staticmethod
    def _score(score):
        if score[0] == 'cp':
            return math.tanh(float(score[1]) / 1000)
        else:
            assert score[0] == 'mate'
            return -1 if score[1].startswith('-') else 1

    def go(self, **limits):
        self.write('go {}\n'.format(' '.join(str(item) for key_value in limits.items() for item in key_value)))
        multipv = {}
        for line in self.read('bestmove'):
            items = line.split()
            if not items:
                continue
            elif items[0] == 'info' and len(items) > 1 and items[1] != 'string' and 'score' in items:
                key = None
                values = []
                info = {}
                for i in items[1:] + ['']:
                    if not i or i in self.KEYWORDS:
                        if key:
                            if values and not issubclass(self.KEYWORDS[key], Iterable):
                                values = values[0]
                            info[key] = self.KEYWORDS[key](values)
                        key = i
                        values = []
                    else:
                        values.append(i)
                info['score'] = self._score(info['score'])
                multipv[info.get('multipv', 1)] = info
        return multipv

    def stop(self):
        self.write('stop\n')

    def read(self, keyword):
        output = []
        while True:
            line = self.process.stdout.readline()
            if not line and self.process.poll() is not None:
                break
            output.append(line)
            if line.startswith(keyword):
                break
        return output


if __name__ == '__main__':
    import sys
    e = Engine(sys.argv[1:])
    e.newgame()
    e.position()
    search_result = e.go(depth=10)
    print(search_result)
