"""
    This script contains classes for antenna arrays

    This script requires that `numpy` be installed within the Python
    environment you are running this script in.

    This file can be imported as a module and contains the following
    class:

    * Antenna

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


class Antenna:
    def __init__(self, x, y, amplitude=1, phase=0):
        self.x = x
        self.y = y
        self.amplitude = amplitude
        self.phase = phase
