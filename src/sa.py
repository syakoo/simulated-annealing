from typing import Callable, TypeVar, List
import math
import random as rd

from .logger import logger

T = TypeVar("T")


class SimulatedAnnealing:
    """
    Simulated Annealingのクラス
    """

    def __init__(self, cooling_schedule: Callable[[int], float], discrete_state_set: List[T], energy_function: Callable[[T], float], n_per: int, t_end: float):
        """インスタンス

        Args:
            cooling_schedule (Callable[[int], float]): クーリング関数
            discrete_state_set (List[T]): 離散状態集合
            energy_function (Callable[[T], float]): エネルギー関数
            n_per (int): nの最大回数
            t_end (float): クーリングエンド T(end)
        """
        self._cooling_schedule = cooling_schedule
        self._discrete_state_set = discrete_state_set
        self._energy_function = energy_function
        self._n_per = n_per
        self._t_end = t_end

    @staticmethod
    def get_sample_cooling_schedule(c: int) -> Callable[[int], float]:
        """クーリング関数の例を取得する

        Args:
            c (int): 調整用(最大値)

        Returns:
            Callable[[int], float]: クーリング関数
        """
        return lambda t: c / (t + 1)

    def _generate_perturbation(self, x: T) -> T:
        """摂動を求める関数。
        離散状態によると思うので、デフォルトでは現在位置付近のデータを返す。

        Args:
            x (T): 現在の離散状態

        Returns:
            T: xの摂動
        """
        if x not in self._discrete_state_set:
            # xがそもそもない場合はエラーを投げる
            raise KeyError(f"A discrete state {x} is not in X.")

        index_x = self._discrete_state_set.index(x)
        return self._discrete_state_set[(index_x + 1) % len(self._discrete_state_set)]

    def _is_acceptable(self, curr: T, next: T, cooling_iter: int) -> bool:
        """求めた摂動を受理するか決定する関数。

        Args:
            curr (T): 現在の離散状態 
            next (T): currの摂動
            cooling_iter (int): クーリングの状態

        Returns:
            bool: nextを受理するかどうか
        """
        e_curr = self._energy_function(curr)
        e_next = self._energy_function(next)

        if e_next <= e_curr:
            return True

        prob = math.exp(-(e_next - e_curr) /
                        self._cooling_schedule(cooling_iter))
        rand = rd.random()
        if rand <= prob:
            return True

        return False

    def iter_log(self, x: T):
        """毎イテレーションで呼び出されるログ用の関数。

        Args:
            x (T): 離散状態
        """
        pass

    def exec(self, x_start: T or None = None) -> T:
        """メインの実行

        Args:
            x_start (T): 初期値

        Returns:
            T: アルゴリズムによる解答
        """
        logger.info(f"Start exec(). x_start = {x_start}")
        t = 0
        x = x_start if not x_start is None else self._discrete_state_set[0]

        while self._cooling_schedule(t) >= self._t_end:
            self.iter_log(x)
            for _ in range(self._n_per):
                x_next: T = self._generate_perturbation(x)
                if self._is_acceptable(x, x_next, t):
                    logger.debug(f"Accept a new state. new_x = {x_next}")
                    x = x_next

            t += 1
        logger.info(f"End exec(). x = {x}, E(x) = {self._energy_function(x)}.")

        return x
