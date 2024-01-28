import arviz as az
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    c8 = az.load_arviz_data("centered_eight")
    nc8 = az.load_arviz_data("non_centered_eight")

    # Exercitiul 1
    print("nr. de lanturi pentru modelul centrat este:", len(c8["posterior"]["chain"]))
    print(
        "marimea totala a esantionului pentru modelul centrat este:",
        len(c8["posterior"]["draw"]) * len(c8["posterior"]["chain"]),
    )
    az.plot_trace(c8, divergences="top")
    plt.show()

    print(
        "nr. de lanturi pentru modelul necentrat este:", len(c8["posterior"]["chain"])
    )
    print(
        "marimea totala a esantionului pentru modelul necentrat este:",
        len(c8["posterior"]["draw"]) * len(c8["posterior"]["chain"]),
    )
    az.plot_trace(nc8, divergences="top")
    plt.show()

    # Exercitiul 2
    print(az.rhat(c8, var_names=["mu", "tau"]))
    print(az.rhat(nc8, var_names=["mu", "tau"]))
    print(az.plot_autocorr(nc8, var_names=["mu", "tau"]))
    print(az.plot_autocorr(c8, var_names=["mu", "tau"]))

    # Exercitiul 3
    print(
        "nr. de divergente pt. modelul centrat este:",
        c8.sample_stats.diverging.sum().item(),
    )
    print(
        "nr. de divergente pt. modelul necentrat este:",
        nc8.sample_stats.diverging.sum().item(),
    )
    az.plot_pair(c8, var_names=["mu", "tau"], divergences=True)
    az.plot_parallel(c8, var_names=["mu", "tau"])

    az.plot_pair(nc8, var_names=["mu", "tau"], divergences=True)
    az.plot_parallel(nc8, var_names=["mu", "tau"])
    plt.show()
