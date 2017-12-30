def error_rate(truth, prediction):
    nb_errors = 0
    dim = len(truth)

    for i in range(dim):
        if prediction[i] != truth[i]:
            nb_errors = nb_errors + 1

    return 1 - (dim - nb_errors) / dim
