import random as rd

from src.tsp_sa import TSP_SA
from src.ksp_sa import KSP_SA


def exec_tsp():
    cs = TSP_SA.get_sample_cooling_schedule(120)
    n_per = 1000
    t_end = 5

    nodes = [(rd.randint(0, 100), rd.randint(0, 100)) for _ in range(20)]
    Model = TSP_SA(cs, nodes, n_per, t_end)
    res = Model.exec()
    value = Model._energy_function(res)

    print(nodes, res, value)
    Model.draw_graph(res)
    Model.output_result_graph()


def exec_ksp():
    cs = KSP_SA.get_sample_cooling_schedule(120)
    n_per = 1000
    t_end = 5

    items = [(rd.randint(10, 100), rd.randint(10, 100)) for _ in range(40)]
    c = rd.randint(400, 800)
    Model = KSP_SA(cs, items, c, n_per, t_end)
    res = Model.exec()
    value = Model._energy_function(res)

    print("items:", items, sep="\n")
    print("c:", c, sep="\n")
    print("res:", res, sep="\n")
    print("value:", value, sep="\n")
    Model.output_result_graph()


def main():
    # exec_tsp()
    exec_ksp()


if __name__ == "__main__":
    main()
