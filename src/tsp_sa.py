from typing import Callable, List, Tuple
from itertools import permutations
import random as rd

import networkx as nx
import matplotlib.pyplot as plt

from .sa import SimulatedAnnealing


class TSP_SA(SimulatedAnnealing):
    def __init__(self, cooling_schedule: Callable[[int], float], route: List[List[int]], n_per: int, t_end: float):
        """TSP用のSA

        Args:
            cooling_schedule (Callable[[int], float]): クーリング関数
            route (List[List[int]]): ルート
            n_per (int): nの最大値
            t_end (float): tの最大値 E(i)
        """
        dis_set, e_func = TSP_SA.route2dataset(route)
        self.routes = route
        self.discrete_state_set = dis_set
        self.energy_function = e_func
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
        discrete_state_set = [list(l) for l in permutations(range_list)]

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
        x1, x2 = rd.sample(list(range(0, len(x))), 2)

        x[x1], x[x2] = x[x2], x[x1]
        return x

    def draw_graph(self, x: List[int]):
        """ルートを画像として出力する

        Args:
            x (List[int]): 離散状態
        """
        G = nx.Graph()
        G.add_nodes_from(list(range(len(self._discrete_state_set[0]))))
        edges = [(x[i], x[(i+1) % len(x)], self.routes[i][(i+1) % len(x)])
                 for i in range(len(x))]
        G.add_weighted_edges_from(edges, weight="weight")

        flg = plt.figure()
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos, node_size=100)
        nx.draw_networkx_edges(G, pos, width=1)

        plt.axis("off")
        # plt.show()
        flg.savefig("img.png")
