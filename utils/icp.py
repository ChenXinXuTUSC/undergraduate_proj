import numpy as np
import open3d as o3d

from . import colorlog
from .colorlog import *
from . import tools
from .tools import *

def early_terminate(curr_res, prev_res):
    relative_fitness_gain = curr_res.fitness / prev_res.fitness - 1.0
    return relative_fitness_gain < 1e-2

def ICP_exact_match(
        srcpts: np.ndarray,
        dstpts: np.ndarray,
        T: np.ndarray,
        max_corresponding_dist, max_iter_num
    ):
    prev_res = o3d.pipelines.registration.evaluate_registration(
        srcpts, dstpts, max_corresponding_dist, T
    )
    curr_res = prev_res

    for _ in range(max_iter_num):

