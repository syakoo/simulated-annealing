import random as rd

from src.tsp_sa import TSP_SA
from src.ksp_sa import KSP_SA
from src.gdp_sa import GDP_SA


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


def exec_gdp():
    cs = GDP_SA.get_sample_cooling_schedule(120)
    n_per = 1000
    t_end = 5
    G_n = 128
    G_k = 64
    G = []
    received_codeword = []

    for k in range(G_k):
        vs = [0 for _ in range(G_k)]
        vs[k] = 1
        G.append(vs)

    for k in range(G_k):
        for _ in range(G_n-G_k):
            G[k].append(rd.randint(0, 1))

    for _ in range(G_n):
        received_codeword.append(rd.randint(0, 1))

    Model = GDP_SA(cs, G, received_codeword, n_per, t_end)
    res = Model.exec()
    value = Model._energy_function(res)

    print("G:", G, sep="\n")
    print("received_codeword:", received_codeword, sep="\n")
    print("res:", res, sep="\n")
    print("value:", value, sep="\n")
    Model.output_result_graph()


def main():
    # exec_tsp()
    # exec_ksp()
    exec_gdp()


if __name__ == "__main__":
    main()
