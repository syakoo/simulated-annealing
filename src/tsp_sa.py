from itertools import permutations
import math
import random as rd
from typing import Callable, List, Tuple

import networkx as nx
import matplotlib.pyplot as plt

from .sa import SimulatedAnnealing
from .logger import logger


class TSP_SA(SimulatedAnnealing):
    def __init__(self, cooling_schedule: Callable[[int], float], nodes: List[Tuple[int, int]], n_per: int, t_end: float):
        """TSP用のSA

        Args:
            cooling_schedule (Callable[[int], float]): クーリング関数
            nodes (List[Tuple[int, int]]): ノードの集合
            n_per (int): nの最大値
            t_end (float): tの最大値 E(i)
        """
        dis_set, e_func = TSP_SA.nodes2dataset(nodes)
        self.nodes = nodes
        self.discrete_state_set = dis_set
        self.energy_function = e_func
        self._result_values = []
        super().__init__(cooling_schedule, dis_set, e_func, n_per, t_end)

    @staticmethod
    def seek_distance_nodes(node_a: Tuple[int, int], node_b: Tuple[int, int]) -> float:
        """二つのノードの距離を求める関数

        Args:
            node_a (Tuple[int, int]): ノードA
            node_b (Tuple[int, int]): ノードB

        Returns:
            float: 距離
        """
        return math.sqrt((node_a[0] - node_b[0]) ** 2 + (node_a[1] - node_b[1]) ** 2)

    @staticmethod
    def nodes2dataset(nodes: List[Tuple[int, int]]) -> Tuple[List[List[int]], Callable[[List[int]], float]]:
        """盤面情報から離散状態集合とエネルギー関数を出力する関数

        Args:
            nodes (List[Tuple[int, int]]): 盤面情報。ノードの情報

        Returns:
            Tuple[List[List[int]], Callable[[List[int]], float]]: 離散状態集合とエネルギー関数
        """
        range_list = list(range(len(nodes)))
        # discrete_state_set = [list(l) for l in permutations(range_list)]
        discrete_state_set = [range_list] # プログラム内で使わないので一つだけ用意する

        def energy_function(x: List[int]) -> float:
            result = 0.0
            for i in range(len(x)):
                result += TSP_SA.seek_distance_nodes(nodes[x[i]],
                                                     nodes[x[(i+1) % len(x)]])

            return result

        return discrete_state_set, energy_function

    def _generate_perturbation(self, x: List[int]) -> List[int]:
        """TSPにおける摂動を求める

        Args:
            x (List[int]): 離散状態

        Returns:
            List[int]: xに対する摂動
        """
        x1, x2 = rd.sample(list(range(1, len(x))), 2)
        result = x[:]

        result[x1], result[x2] = x[x2], x[x1]
        return result

    def draw_graph(self, x: List[int], title: str = ""):
        """ルートを画像として出力する

        Args:
            x (List[int]): 離散状態
        """
        G = nx.Graph()
        G.add_nodes_from(list(range(len(self.nodes))))
        edges = [(x[i], x[(i+1) % len(x)], {})
                 for i in range(len(x))]
        G.add_weighted_edges_from(edges, weight="weight")
        pos = dict([(xi, self.nodes[xi]) for xi in x])

        fig = plt.figure()
        nx.draw_networkx(G, pos, node_color="#bbb")

        fig.suptitle(title)
        plt.axis("off")

        if title == "":
            fig.savefig("./images/img.png")
        else:
            fig.savefig(f"./images/{title}.png")

    def iter_log(self, x: List[int], t: int):
        logger.info(f"iter: t = {t}")
        distance = self.energy_function(x)
        self._result_values.append(distance)

    def output_result_graph(self):
        fig = plt.figure()
        plt.plot(list(
            range(len(self._result_values))), self._result_values, marker="o")
        plt.xlabel("t")
        plt.ylabel("distance")
        fig.savefig("./images/graph.png")
