import random as rd
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt

from .sa import SimulatedAnnealing
from .logger import logger


class KSP_SA(SimulatedAnnealing):
    def __init__(self, cooling_schedule: Callable[[int], float], items: List[Tuple[int, int]], c: int, n_per: int, t_end: float):
        """KSP用のSA

        Args:
            cooling_schedule (Callable[[int], float]): クーリング関数
            items (List[Tuple[int, int]]): 品物, (価値 pi, 容積 ci)
            c (int): ナップサックの容量
            n_per (int): n の最大値
            t_end (float): t の最大値 E(i)
        """
        dis_set, e_func = KSP_SA.items2dataset(items, c)
        self.discrete_state_set = dis_set
        self.energy_function = e_func
        self._result_values = []
        super().__init__(cooling_schedule, dis_set, e_func, n_per, t_end)

    @staticmethod
    def items2dataset(items: List[Tuple[int, int]], c: int) -> Tuple[List[List[int]], Callable[[List[int]], int]]:
        """品物から離散状態集合とエネルギー関数を出力する

        Args:
            items (List[Tuple[int, int]]): 品物.
            c (int): 容量.

        Returns:
            Tuple[List[List[int]], Callable[[List[int]], int]]: 離散状態集合とエネルギー関数
        """
        # 初期値としてしか使わない
        idx_list = [0 for _ in range(len(items))]
        discrete_state_set = [idx_list]

        def energy_function(x: List[int]) -> int:
            c_sum, p_sum = 0, 0
            for i, xi in enumerate(x):
                p_sum += xi*items[i][0]
                c_sum += xi*items[i][1]
            # 容量より多い場合は、0を返す
            if c_sum > c:
                return 0

            return - p_sum

        return discrete_state_set, energy_function

    def _generate_perturbation(self, x: List[int]) -> List[int]:
        """KSPにおける摂動を求める

        Args:
            x (List[int]): 離散状態

        Returns:
            List[int]: xに対する摂動
        """
        idx_x = rd.randint(0, len(x) - 1)
        result = x[:]

        result[idx_x] = (result[idx_x] + 1) % 2
        return result

    def iter_log(self, x: List[int], t: int):
        logger.info(f"iter: t = {t}")
        distance = self.energy_function(x)
        self._result_values.append(distance)

    def output_result_graph(self):
        fig = plt.figure()
        plt.plot(list(
            range(len(self._result_values))), self._result_values, marker="o")
        plt.xlabel("t")
        plt.ylabel("value")
        fig.savefig("./images/ksp_graph.png")
