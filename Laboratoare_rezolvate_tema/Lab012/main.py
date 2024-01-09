import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def calculate_pi(N):
    x, y = np.random.uniform(-1, 1, size=(2, N))
    inside = (x ** 2 + y ** 2) <= 1
    pi = inside.sum() * 4 / N
    error = abs((pi - np.pi) / pi) * 100
    return error


def posterior_grid(grid_points=50, heads=6, tails=9, prior=None):
    grid = np.linspace(0, 1, grid_points)

    if prior is None:
        prior = np.repeat(1 / grid_points, grid_points)

    likelihood = stats.binom.pmf(heads, heads + tails, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()

    return grid, posterior


def metropolis(func, draws=10000):
    trace = np.zeros(draws)
    old_x = 0.5
    old_prob = func.pdf(old_x)
    delta = np.random.normal(0, 0.5, draws)

    for i in range(draws):
        new_x = old_x + delta[i]
        new_prob = func.pdf(new_x)
        acceptance = new_prob / old_prob

        if acceptance >= np.random.random():
            trace[i] = new_x
            old_x = new_x
            old_prob = new_prob
        else:
            trace[i] = old_x

    return trace


def plot_posterior(grid, posterior, title):
    plt.plot(grid, posterior, 'o-')
    plt.title(title)
    plt.yticks([])
    plt.xlabel('θ')
    plt.show()


def plot_pi_estimation():
    values_N = [100, 1000, 10000]
    mean_err = []
    std_err = []

    for value in values_N:
        errors = [calculate_pi(value) for _ in range(100)]
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        mean_err.append(mean_error)
        std_err.append(std_error)

    for i, value in enumerate(values_N, 1):
        print(f'Mean for N = {value}: {mean_err[i - 1]}, std_dev: {std_err[i - 1]}')

    plt.figure(figsize=(10, 6))
    plt.errorbar(values_N, mean_err, yerr=std_err, fmt='o-', capsize=5, color='blue',
                 label='Estimation with Error Bars')
    plt.xscale('log')
    plt.xlabel('Number of Points (N)', fontsize=12)
    plt.ylabel('Error (%)', fontsize=12)
    plt.title('Estimation of π with Error Bars', fontsize=14)
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.show()


def plot_beta_distributions():
    plt.figure(figsize=(10, 5))
    beta_params = [(1, 1), (20, 20), (1, 4)]

    for i, item in enumerate(beta_params, 1):
        plt.subplot(1, 3, i)
        func = stats.beta(item[0], item[1])
        trace = metropolis(func=func)
        x = np.linspace(0.01, .99, 100)
        y = func.pdf(x)
        y_true = func.pdf(x)
        plt.plot(x, y_true, label=f'True distribution - {item}', lw=3)
        plt.hist(trace[trace > 0], bins=25, density=True, alpha=0.5, label=f'Estimated distribution - {item}')
        plt.xlim(0, 1)
        plt.xlabel('x')
        plt.ylabel('pdf(x)')
        plt.yticks([])

    plt.legend()
    plt.tight_layout()
    plt.show()


def compare_beta_distributions():
    beta_params = [(1, 1), (20, 20), (1, 4)]
    plt.figure(figsize=(15, 8))

    for item in beta_params:
        data = np.repeat([0, 1], (item[1], item[0]))
        points = 10
        h = data.sum()
        t = len(data) - h
        grid, posterior = posterior_grid(points, h, t)

        plt.subplot(1, 2, 1)
        plt.plot(grid, posterior, 'o-', label=f'Posterior - {item}')
        plt.title('Grid Computing')
        plt.xlabel('θ')
        plt.ylabel('pdf(θ)')
        plt.legend()

        func = stats.beta(item[0], item[1])
        trace = metropolis(func=func)

        plt.subplot(1, 2, 2)
        x = np.linspace(0.01, .99, 100)
        y = func.pdf(x)
        plt.xlim(0, 1)
        plt.plot(x, y, 'C1-', lw=3, label='True distribution')
        plt.hist(trace[trace > 0], bins=25, density=True, alpha=0.5, label='Estimated distribution')

    plt.subplot(1, 2, 1)
    plt.tight_layout()
    plt.show()


def main():
    data = np.repeat([0, 1], (15, 25))
    points = 10
    h = data.sum()
    t = len(data) - h
    grid, posterior = posterior_grid(points, h, t)
    plot_posterior(grid, posterior, f'heads = {h}, tails = {t} (Uniform Prior)')

    prior_new1 = (grid <= 0.5).astype(int)
    grid, posterior = posterior_grid(points, h, t, prior=prior_new1)
    plot_posterior(grid, posterior, f'heads = {h}, tails = {t} (Prior: grid <= 0.5)')

    prior_new2 = np.abs(grid - 0.5)
    grid, posterior = posterior_grid(points, h, t, prior=prior_new2)
    plot_posterior(grid, posterior, f'heads = {h}, tails = {t} (Prior: abs(grid - 0.5)')

    plot_pi_estimation()

    plot_beta_distributions()

    compare_beta_distributions()


if __name__ == '__main__':
    main()
