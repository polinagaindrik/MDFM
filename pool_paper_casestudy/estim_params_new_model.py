import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def get_initial_guesses(omega, K, n, T_max=3000, n_samples=200):
    """
    Fit simplified death-term candidates against the current Hill-function
    death curve, to get warm-start initial guesses for each alternative
    before refitting to real data.

    omega, K, n : your current fitted Hill-function parameters
    T_max       : upper end of the T range to sample (use your actual
                  observed/simulated T range, e.g. ~3000)
    n_samples   : number of points to sample the Hill curve at

    Returns a dict of initial guesses for each candidate model.
    """
    T_sample = np.linspace(0, T_max, n_samples)
    f_hill = omega * np.abs(T_sample) ** n / (K ** n + np.abs(T_sample) ** n)

    guesses = {}

    # 1. Linear: f(T) = k1 * T
    def f_lin(T, k1):
        return k1 * T
    p_lin, _ = curve_fit(f_lin, T_sample, f_hill, p0=[omega / K])
    guesses["linear"] = {"k1": p_lin[0]}

    # 2. Saturating exponential: f(T) = omega2 * (1 - exp(-T/K2))
    def f_exp(T, omega2, K2):
        return omega2 * (1 - np.exp(-T / K2))
    p_exp, _ = curve_fit(f_exp, T_sample, f_hill, p0=[omega, K / np.log(2)], maxfev=5000)
    guesses["exp_saturating"] = {"omega2": p_exp[0], "K2": p_exp[1]}

    # 3. Michaelis-Menten (Hill with n fixed to 1): f(T) = omega3 * T / (K3 + T)
    def f_mm(T, omega3, K3):
        return omega3 * T / (K3 + T)
    p_mm, _ = curve_fit(f_mm, T_sample, f_hill, p0=[omega, K], maxfev=5000)
    guesses["michaelis_menten"] = {"omega3": p_mm[0], "K3": p_mm[1]}

    return guesses


if __name__ == "__main__":
    omega = 1.0588212004920692
    K = 4189.073636034137
    n = 0.38321670119758156

    guesses = get_initial_guesses(omega, K, n, T_max=3000)
    for model, params in guesses.items():
        print(f"{model}: {params}")


    T_sample = np.linspace(0, 3000, 200)
    f_hill = omega * T_sample**n / (K**n + T_sample**n)
    f_lin = 0.00022282072264787102 * T_sample
    f_exp = 0.45462171090600184 * (1 - np.exp(-T_sample / 376.602341918351))
    f_mm = 0.5108470797227876 * T_sample / (T_sample + 254.79790790361722)

    plt.plot(T_sample, f_hill, label="Hill (current)", lw=2)
    plt.plot(T_sample, f_lin, "--", label="linear")
    plt.plot(T_sample, f_exp, "--", label="exp saturating")
    plt.plot(T_sample, f_mm, "--", label="Michaelis-Menten")
    plt.legend(); plt.xlabel("T"); plt.ylabel("death rate term")
    plt.savefig("death_term_comparison.png", dpi=200)