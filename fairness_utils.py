def fairness_violation(solution):
    """
    Measures how sensitive prediction is to protected flips
    """
    base_pred = predictive_model.model.predict(
        solution.reshape(1, -1)
    )[0]

    deltas = []
    for cf in fairness_solutions[:10]:  # sample
        cf = repair_solution(cf, prefix_indices)
        pred_cf = predictive_model.model.predict(
            cf.reshape(1, -1)
        )[0]
        deltas.append(abs(base_pred - pred_cf))

    return np.mean(deltas)
