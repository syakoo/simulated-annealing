import math
import random as rd
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt

from .sa import SimulatedAnnealing
from .logger import logger


class GDP_SA(SimulatedAnnealing):
    def __init__(self, cooling_schedule: Callable[[int], float], G: List[List[int]], codeword: List[int], n_per: int, t_end: float):
        """GDP用のSA

        Args:
            cooling_schedule (Callable[[int], float]): クーリング関数
            G (List[List[int]]): 符号の生成行列
            codeword (List[int]): 受信した符号語
            n_per (int): nの最大値
            t_end (float): tの最大値 E(i)
        """
        dis_set, e_func = GDP_SA.codes2dataset(G, codeword)
        self.G = G
        self.discrete_state_set = dis_set
        self.energy_function = e_func
        self._result_values = []
        super().__init__(cooling_schedule, dis_set, e_func, n_per, t_end)

    @staticmethod
    def seek_distance_codes(c1: List[int], c2: List[int]) -> int:
        """二つの符号のハミング距離を求める関数

        Args:
            c1 (List[int]): 符号1
            c2 (List[int]): 符号2

        Returns:
            int: ハミング距離
        """
        result = 0
        for i in range(len(c1)):
            if c1[i] != c2[i]:
                result += 1

        return result

    @staticmethod
    def codes2dataset(G: List[List[int]], codeword: List[int]) -> Tuple[List[List[int]], Callable[[List[int]], int]]:
        """符号から離散状態集合とエネルギー関数を出力する関数

        Args:
            nodes (List[Tuple[int, int]]): 符号
            codeword (List[int]): 受信した符号語

        Returns:
            Tuple[List[List[int]], Callable[[List[int]], int]]: 離散状態集合とエネルギー関数
        """
        range_list = [0 for _ in range(len(G))]
        discrete_state_set = [range_list]  # プログラム内で使わないので一つだけ用意する(初期値のみ)

        def energy_function(x: List[int]) -> int:
            code = [0 for _ in range(len(G[0]))]
            for i, xi in enumerate(x):
                if xi == 0:
                    continue

                for j in range(len(code)):
                    code[j] = (code[j] + G[i][j]) % 2

            # 影響を出すために10倍する
            return GDP_SA.seek_distance_codes(code, codeword) * 10

        return discrete_state_set, energy_function

    def _generate_perturbation(self, x: List[int]) -> List[int]:
        """GDPにおける摂動を求める

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
        plt.ylabel("Hamming distance(*10)")
        fig.savefig("./images/gdp_graph.png")
