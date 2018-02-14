
from oracle.frameworks.scrabble_interface import ScrabbleInterface
from oracle.db import *


building = 'ap_m'

scrabble = ScrabbleInterface(building)
scrabble.learn_auto()
