# main/helpers.py



def model_selection_prior(hyperparameter: float, k: int, p: int):
    return ((hyperparameter) ** k) * ((1 - hyperparameter) ** (p - k))

def deletion(gamma: tuple[int]) -> list[tuple[int]]:
    return [tuple(list(gamma[:i]) + [0] + list(gamma[i + 1:])) for i, k in enumerate(gamma) if k == 1]

def addition(gamma: tuple[int]) -> list[tuple[int]]:
    return [tuple(list(gamma[:i]) + [1] + list(gamma[i + 1:])) for i, k in enumerate(gamma) if k == 0]

def replacement(gamma: tuple[int]) -> list[tuple[int]]:
    gamma_replacement = []

    for i, k in enumerate(gamma):
        for j, l in enumerate(gamma):
            if k == 1 and l == 0 and i != j:
                gamma_i = list(gamma)
                gamma_i[i] = 0
                gamma_i[j] = 1
                gamma_replacement.append(tuple(gamma_i))

    return gamma_replacement

def nbd(gamma):
    return addition(gamma), replacement(gamma), deletion(gamma)




