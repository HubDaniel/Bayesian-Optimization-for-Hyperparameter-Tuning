# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 15:23:52 2022

@author: @author: reference.  https://github.com/fmfn/BayesianOptimization
"""

class Events:
    OPTIMIZATION_START = 'optimization:start'
    OPTIMIZATION_STEP = 'optimization:step'
    OPTIMIZATION_END = 'optimization:end'


DEFAULT_EVENTS = [
    Events.OPTIMIZATION_START,
    Events.OPTIMIZATION_STEP,
    Events.OPTIMIZATION_END,
]