from typing import Callable, List, Tuple
from itertools import permutations
import random as rd

from .sa import SimulatedAnnealing


class TSP_SA(SimulatedAnnealing):
    def __init__(self, cooling_schedule: Callable[[int], float], route: List[List[int]], n_per: int, t_end: float):
        dis_set, e_func = TSP_SA.route2dataset(route)
        super().__init__(cooling_schedule, dis_set, e_func, n_per, t_end)

    @staticmethod
    def route2dataset(route: List[List[int]]) -> Tuple[List[List[int]], Callable[[List[int]], float]]:
        """盤面情報から離散状態集合とエネルギー関数を出力する関数

        Args:
            route (List[List[int]]): 盤面情報。正方行列

        Returns:
            Tuple[List[List[int]], Callable[[List[int]], float]]: 離散状態集合とエネルギー関数
        """
        range_list = list(range(len(route)))
        discrete_state_set = list(permutations(range_list))

        def energy_function(x: List[int]) -> float:
            result = 0
            for i in range(len(x)):
                result += route[x[i]][x[(i+1) % len(x)]]

            return result

        return discrete_state_set, energy_function

    def _generate_perturbation(self, x: List[int]) -> List[int]:
        """TSPにおける摂動を求める

        Args:
            x (List[int]): 離散状態

        Returns:
            List[int]: xに対する摂動
        """
        x1, x2 = [rd.randint(0, len(x)) for _ in range(2)]
        if x1 == x2:
            x2 = (x1 + x2) % len(x)

        x[x1], x[x2] = x[x2], x[x1]
        return x
