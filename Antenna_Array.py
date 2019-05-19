"""
    This script contains classes for antenna arrays

    This script requires that `numpy` be installed within the Python
    environment you are running this script in.

    This file can be imported as a module and contains the following
    class:

    * Antenna_Array

    ----------
    Antarray - Antenna Array Analysis Module
    Copyright (C) 2018 - 2019  Zhengyu Peng
    E-mail: zpeng.me@gmail.com
    Website: https://zpeng.me

    `                      `
    -:.                  -#:
    -//:.              -###:
    -////:.          -#####:
    -/:.://:.      -###++##:
    ..   `://:-  -###+. :##:
           `:/+####+.   :##:
    .::::::::/+###.     :##:
    .////-----+##:    `:###:
     `-//:.   :##:  `:###/.
       `-//:. :##:`:###/.
         `-//:+######/.
           `-/+####/.
             `+##+.
              :##:
              :##:
              :##:
              :##:
              :##:
               .+:

"""

import numpy as np
from antarray import Antenna


class Antenna_Array:
    def __init__(self, antenna_list):
        self.size = len(antenna_list)
        self.x = np.zeros(self.size)
        self.y = np.zeros(self.size)
        self.phase = np.zeros(self.size)
        self.amplitude = np.zeros(self.size)

        for ant_idx, ant in enumerate(antenna_list):
            self.x[ant_idx] = ant.x
            self.y[ant_idx] = ant.y
            self.phase[ant_idx] = ant.phase
            self.amplitude[ant_idx] = ant.amplitude
            