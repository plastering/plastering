from oracle.frameworks.zodiac_interface import ZodiacInterface
from oracle.db import *


building = 'ap_m'

zodiac = ZodiacInterface(building)
zodiac.learn_auto()
