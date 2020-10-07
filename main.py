import random as rd

from src.tsp_sa import TSP_SA


def main():
    cs = TSP_SA.get_sample_cooling_schedule(100)
    n_per = 1000
    t_end = 10

    nodes = [(rd.randint(0, 1000), rd.randint(0, 1000)) for _ in range(10)]
    Model = TSP_SA(cs, nodes, n_per, t_end)
    res = Model.exec()
    value = Model._energy_function(res)

    print(nodes, res, value)
    Model.draw_graph(res)


if __name__ == "__main__":
    main()
