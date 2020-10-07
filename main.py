from src.tsp_sa import TSP_SA

test_case = [
    ([[0, 6, 5, 5], [6, 0, 7, 4], [5, 7, 0, 3], [5, 4, 3, 0]], [0, 1, 3, 2]),
    ([[0, 8, 3, 5], [8, 0, 7, 4], [3, 7, 0, 4], [5, 4, 4, 0]], [0, 1, 3, 2])
]


def main():
    cs = TSP_SA.get_sample_cooling_schedule(30)
    n_per = 100
    t_end = 50

    for routes, ans in test_case:
        Model = TSP_SA(cs, routes, n_per, t_end)
        res = Model.exec()
        value = Model._energy_function(res)
        ans_value = Model._energy_function(ans)

        print(res, value, ans_value)
        Model.draw_graph(res)


if __name__ == "__main__":
    main()
