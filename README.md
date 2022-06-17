# Chess variant MCTS search

This project provides an MCTS/UCT search based on [brilee/python_uct](https://github.com/brilee/python_uct) on top of UCI chess variant engines. This can be useful to provide features that alpha-beta search engines can not, such as providing a tree with a balance between exploitation and exploration.

Possible use cases can be the generation of opening books, identification of critical lines, and the like.

The recommended engine to use with this project is [Fairy-Stockfish](https://github.com/ianfab/Fairy-Stockfish), but in principle any UCI chess variant engine could be used.

## Setup
The project requires at least python3.5 as well as the dependencies from the `requirements.txt`. Install them using
```
pip3 install -r requirements.txt
```

## Usage
A simple example of running the script is:
```
python3 mcts.py --engine fairy-stockfish.exe --variant chess --rollouts 100
```
Run the script with `--help` to get help on the supported parameters.
