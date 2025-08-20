import numpy as np
import math
from scipy.optimize import minimize


__all__ = ["DecoyStateFinite", "DecoyStateAsymptotic"]


def h(x):
    if x in [0, 1]:
        return 0
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)


def expected_detection_rate(transmittance, k, Y_0):
    r"""
    transmittance: overall transmittance
    k: intensity $k \in \{\mu_1, \mu_2, \mu_3\}$
    Y_0:  dark count probability
    """

    transmittance = np.asarray(transmittance)
    gain = 1 - (1 - Y_0) * np.exp(-transmittance * k)
    return gain.mean()


def intensity_bit_error_rate(transmittance, k, Y_0, e_mis, e_0=0.5):
    r"""
    transmittance: overall transmittance
    k: intensity $k \in \{\mu_1, \mu_2, \mu_3\}$
    Y_0:  dark count probability
    e_mis:  error rate due to optical errors
    e_0: error rate of the background
    """
    gain = expected_detection_rate(transmittance, k, Y_0)
    return (e_0 * Y_0 + e_mis * (gain - Y_0)) / gain


def tau_n(mu_1, mu_2, mu_3, p_1, p_2, p_3, n):
    terms = []
    for k, p_k in zip([mu_1, mu_2, mu_3], [p_1, p_2, p_3]):
        term = np.exp(-k) * k**n * p_k / math.factorial(n)
        terms.append(term)
    return np.sum(terms)


def vacuum_events_lower_bound(mu_1, mu_2, mu_3,
                              p_mu1, p_mu2, p_mu3,
                              n_mu2, n_mu3):
    tau_0 = tau_n(mu_1, mu_2, mu_3, p_mu1, p_mu2, p_mu3, 0)
    term1 = tau_0 / (mu_2 - mu_3)
    term2 = mu_2 * np.exp(mu_3) * n_mu3 / p_mu3
    term3 = mu_3 * np.exp(mu_2) * n_mu2 / p_mu2
    res = term1 * (term2 - term3)
    return max(res, 0)


def single_photon_events_lower_bound(mu_1, mu_2, mu_3,
                                     p_mu1, p_mu2, p_mu3,
                                     n_mu_1, n_mu_2, n_mu_3, s_0):
    tau_0 = tau_n(mu_1, mu_2, mu_3, p_mu1, p_mu2, p_mu3, 0)
    tau_1 = tau_n(mu_1, mu_2, mu_3, p_mu1, p_mu2, p_mu3, 1)
    denominator = mu_1 * (mu_2 - mu_3) - (mu_2**2 - mu_3**2)
    term1 = (np.exp(mu_2) * n_mu_2) / p_mu2
    term2 = (np.exp(mu_3) * n_mu_3) / p_mu3
    term3 = ((mu_2**2 - mu_3**2) / (mu_1**2)) * (np.exp(mu_1) / p_mu1 * n_mu_1 - s_0 / tau_0)
    bracketed = term1 - term2 - term3
    coefficient = (mu_1 * tau_1) / denominator
    res = coefficient * bracketed
    return max(res, 0)


def single_photon_errors_upper_bound(mu_1, mu_2, mu_3,
                                     p_mu1, p_mu2, p_mu3,
                                     m_mu2, m_mu3):
    tau_1 = tau_n(mu_1, mu_2, mu_3, p_mu1, p_mu2, p_mu3, 1)
    term1 = tau_1 / (mu_2 - mu_3)
    term2 = np.exp(mu_2) / p_mu2 * m_mu2
    term3 = np.exp(mu_3) / p_mu3 * m_mu3
    res = term1 * (term2 - term3)
    return max(res, 0)


class DecoyStateFinite():
    def __init__(self, mu_k, p_k,
                 n_Xk, n_Zk, m_Xk, m_Zk,
                 f_EC=1.22,
                 eps_sec=1e-10,
                 eps_cor=1e-10,
                 max_phase_error=0.1
                 ):
        """
        Initializes the DecoyState object with observed experimental data.

        Args:
            mu_k (list): A list of three intensity values [mu_1, mu_2, mu_3].
            p_k (list): A list of three probabilities [p_1, p_2, p_3] for choosing each intensity.
            n_Xk (list): Counts in the X basis for each intensity.
            n_Zk (list): Counts in the Z (Z) basis for each intensity.
            m_Xk (list): Error counts in the X basis for each intensity.
            m_Zk (list): Error counts in the Z (Z) basis for each intensity.
            f_EC (float): Error correction inefficiency factor.
            eps_sec (float): Security parameter.
            eps_cor (float): Correctness parameter.
            max_phase_error (float):
        """
        if any(len(arr) != 3 for arr in [mu_k, p_k, n_Xk, n_Zk, m_Xk, m_Zk]):
            raise ValueError("Length of all array-like inputs must be 3.")
        if not all(mu >= 0 for mu in mu_k):
            raise ValueError(f'Intensities must be non-negative. {mu_k=}')
        if not mu_k[1] > mu_k[2]:
            raise ValueError(f'Constraint not met: mu_2 > mu_3. {mu_k[1]=}, {mu_k[2]=}')
        if not mu_k[0] > mu_k[1] + mu_k[2]:
            raise ValueError(f'Constraint not met: mu_1 > mu_2 + mu_3. {mu_k=}')
        if not all(0 < p < 1 for p in p_k):
            raise ValueError(f'Probabilities must be in (0, 1). {p_k=}')
        if not np.isclose(sum(p_k), 1):
            raise ValueError(f'Sum of probabilities must be 1. {p_k=}')

        self.mu_k = mu_k
        self.p_k = p_k
        self.n_Xk = np.asarray(n_Xk)
        self.n_Zk = np.asarray(n_Zk)
        self.m_Xk = np.asarray(m_Xk)
        self.m_Zk = np.asarray(m_Zk)
        self.f_EC = f_EC
        self.eps_sec = eps_sec
        self.eps_cor = eps_cor
        self.max_phase_error = max_phase_error

    def __repr__(self):
        mu_k = ', '.join([f"{x:.2f}" for x in self.mu_k])
        p_k = ', '.join([f"{x:.2f}" for x in self.p_k])
        n_Xk = ', '.join([f"{x:.1e}" for x in self.n_Xk])
        n_Zk = ', '.join([f"{x:.1e}" for x in self.n_Zk])
        m_Xk = ', '.join([f"{x:.1e}" for x in self.m_Xk])
        m_Zk = ', '.join([f"{x:.1e}" for x in self.m_Zk])
        return (f"{self.__class__.__name__}("
                f"mu_k=[{mu_k}], "
                f"p_k=[{p_k}], "
                f"n_Xk=[{n_Xk}], "
                f"n_Zk=[{n_Zk}], "
                f"m_Xk=[{m_Xk}], "
                f"m_Zk=[{m_Zk}], "
                f"f_EC={self.f_EC:.2f}, "
                f"eps_sec={self.eps_sec:.1e}, "
                f"eps_cor={self.eps_cor:.1e})")

    def key_length(self, clip_neg=True, raise_phase_error=False, debug=False):
        n_X_tot = np.sum(self.n_Xk)
        n_Z_tot = np.sum(self.n_Zk)
        m_X_tot = np.sum(self.m_Xk)
        m_Z_tot = np.sum(self.m_Zk)

        # Lower bound on vacuum events (s0)
        s_0_X_low = self._vacuum_events_finite_lower_bound(self.n_Xk[1], self.n_Xk[2], n=n_X_tot)
        s_0_Z_low = self._vacuum_events_finite_lower_bound(self.n_Zk[1], self.n_Zk[2], n=n_Z_tot)

        # Lower bound on single-photon events (s1)
        s_1_X_low = self._single_photon_events_finite_lower_bound(self.n_Xk[0], self.n_Xk[1], self.n_Xk[2], s_0_X_low, n=n_X_tot)
        s_1_Z_low = self._single_photon_events_finite_lower_bound(self.n_Zk[0], self.n_Zk[1], self.n_Zk[2], s_0_Z_low, n=n_Z_tot)

        # Upper bound on single-photon errors in Z basis (nu1)
        nu_1_Z_upper = self._single_photon_errors_finite_upper_bound(self.m_Zk[1], self.m_Zk[2], m=m_Z_tot)

        if s_1_Z_low <= 0 or s_1_Z_low < nu_1_Z_upper:
            return 0

        # Upper bound on the phase error rate of single-photon events (phi)
        phi_Z = self._single_photon_phase_error_finite_upper_bound(s_1_X_low, s_1_Z_low, nu_1_Z_upper)

        if phi_Z > self.max_phase_error:
            if raise_phase_error:
                raise ValueError(f"Phase error {phi_Z=} is greater than {self.max_phase_error=}.")
            return 0

        lambda_EC = self.f_EC * n_X_tot * h((self.m_Xk[0] + self.m_Xk[1]) / n_X_tot)

        if debug:
            print(f"{s_0_X_low=:.0f}, {s_0_Z_low=:.0f}, {s_1_X_low=:.0f}, {s_1_Z_low=:.0f}, {nu_1_Z_upper=:.0f}, {lambda_EC=:.0f}")

        # Privacy amplification terms
        term1 = 6 * np.log2(21 / self.eps_sec)
        term2 = np.log2(2 / self.eps_cor)

        # Final secure key length
        r = s_0_X_low + s_1_X_low * (1 - h(phi_Z)) - lambda_EC - term1 - term2
        return max(r, 0) if clip_neg else r

    @classmethod
    def from_channel_params(cls, mu_k, p_k, p_X, transmittance, N, Y_0=1.7e-6, e_mis=0.033,
                            p_a=None, f_EC=1.22, eps_sec=1e-10, eps_cor=1e-10, max_phase_error=0.1):
        """
        Creates a DecoyState instance from channel parameters.

        Args:
            N (int): Total number of pulses sent.
            transmittance (float | np.arraylike): Overall hannel transmittance (eta).
            Y_0 (float): Dark count probability per pulse.
            e_mis (float): Misalignment error rate.
            mu_k (list): Intensities [mu_1, mu_2, mu_3].
            p_k (list): Probabilities of sending each intensity.
            p_X (float): Probability of choosing the X basis.
            p_a (float, optional): After-pulsing probability. Defaults to None.
        """
        params = dict(mu_k=mu_k, p_k=p_k, f_EC=f_EC, eps_sec=eps_sec, eps_cor=eps_cor, max_phase_error=max_phase_error)
        p_Z = 1 - p_X
        if not (0 < p_X < 1):
            raise ValueError(f'Basis probability must be 0 < p_X < 1. {p_X=}')

        gains = [expected_detection_rate(transmittance, mu, Y_0) for mu in mu_k]
        qbers = [intensity_bit_error_rate(transmittance, mu, Y_0, e_mis) for mu in mu_k]

        if p_a is not None:
            qbers = [q + p_a / 2 for q in qbers]
            gains = [g * (1 + p_a) for g in gains]

        counts = []
        for gain_k, qber_k, p_muk in zip(gains, qbers, p_k):
            n_X = N * p_muk * p_X**2 * gain_k
            m_X = n_X * qber_k
            n_Z = N * p_muk * p_Z**2 * gain_k
            m_Z = n_Z * qber_k
            counts.append([n_X, n_Z, m_X, m_Z])

        counts = dict(zip(["n_Xk", "n_Zk", "m_Xk", "m_Zk"], np.asarray(counts).T))
        return cls(**counts, **params)

    @staticmethod
    def random_channel_params(transmittance, N, Y_0, e_mis, p_a=None, f_EC=1.22, eps_sec=1e-10, eps_cor=1e-10, max_phase_error=0.1, tries=100000, seed=0, debug=False):
        params = dict(transmittance=transmittance, N=N, Y_0=Y_0, e_mis=e_mis, p_a=p_a, f_EC=f_EC, eps_sec=eps_sec, eps_cor=eps_cor, max_phase_error=max_phase_error)

        def random_params(seed=None):
            rng = np.random.RandomState(seed)
            _mu1 = rng.uniform()
            _mu2 = rng.uniform(high=_mu1)
            _mu3 = rng.uniform(high=min(_mu2, _mu1 - _mu2))
            _p1 = rng.uniform()
            _p2 = rng.uniform(high=1 - _p1)
            _pX = rng.uniform(low=0.5)
            return dict(mu_k=[_mu1, _mu2, _mu3], p_k=[_p1, _p2, 1 - _p1 - _p2], p_X=_pX)

        for _i in range(tries):
            rnd = random_params(seed=seed + _i)
            test_decoy_state = DecoyStateFinite.from_channel_params(**rnd, **params)
            key_length = test_decoy_state.key_length()
            if key_length > 0:
                return rnd
        raise RuntimeError("Can't find any set of parameters for positive keyrate.")

    @staticmethod
    def optimize_params(transmittance, N, Y_0, e_mis, p_a=None, f_EC=1.22, eps_sec=1e-10, eps_cor=1e-10, max_phase_error=0.1, initial_guess_tries=100000, mu_1=None, mu_2=None, mu_3=0, p_1=None, p_2=None, p_X=None, seed=0, debug=False):
        params = dict(transmittance=transmittance, N=N, Y_0=Y_0, e_mis=e_mis, p_a=p_a, f_EC=f_EC, eps_sec=eps_sec, eps_cor=eps_cor, max_phase_error=max_phase_error)

        params_order = ['mu_1', 'mu_2', 'mu_3', 'p_1', 'p_2', 'p_X']
        fixed_values = [mu_1, mu_2, mu_3, p_1, p_2, p_X]
        fixed_mask = [p is not None for p in fixed_values]
        free_indices, free_params = zip(*[(i, name) for i, (fixed, name) in enumerate(zip(fixed_mask, params_order)) if not fixed])

        def get_fixed_or_free(x, key):
            if key in free_params:
                return x[free_params.index(key)]
            else:
                return fixed_values[params_order.index(key)]

        def objective(x):
            objective_params = {
                "mu_k": [get_fixed_or_free(x, 'mu_1'), get_fixed_or_free(x, 'mu_2'), get_fixed_or_free(x, 'mu_3')],
                "p_k": [get_fixed_or_free(x, 'p_1'), get_fixed_or_free(x, 'p_2'), 1 - get_fixed_or_free(x, 'p_1') - get_fixed_or_free(x, 'p_2')],
                "p_X": get_fixed_or_free(x, 'p_X'),
            }
            try:
                decoy_state = DecoyStateFinite.from_channel_params(**params, **objective_params)
                return -1 * decoy_state.key_length(clip_neg=False, raise_phase_error=True)
            except ValueError:
                return 0

        def debug_func(x):
            print(x, -1 * objective(x) / N)

        initial_guess = [0.8, 0.15, 0, 0.85, 0.1, 0.95]
        if objective(initial_guess) <= 0:
            _g = DecoyStateFinite.random_channel_params(**params, debug=debug, seed=seed, tries=initial_guess_tries)
            initial_guess = (_g['mu_k'][0], _g['mu_k'][1], _g['mu_k'][2], _g['p_k'][0], _g['p_k'][1], _g['p_X'])


        x0 = [initial_guess[i] for i in free_indices]
        bounds = [(0, 1)] * len(x0)

        result = minimize(objective, x0, bounds=bounds,
                        options={"fatol": 1e-9, "xatol": 1e-9, "disp": debug, "maxiter": 1000},
                        method="Nelder-Mead",
                        callback=debug_func if debug else None
                        )
        optimized_params = {
            "mu_k": (float(get_fixed_or_free(result.x, 'mu_1')),
                     float(get_fixed_or_free(result.x, 'mu_2')),
                     float(get_fixed_or_free(result.x, 'mu_3')),
                    ),
            "p_k": (float(get_fixed_or_free(result.x, 'p_1')),
                    float(get_fixed_or_free(result.x, 'p_2')),
                    1 - float(get_fixed_or_free(result.x, 'p_1')) - float(get_fixed_or_free(result.x, 'p_2'))
                    ),
            "p_X": float(get_fixed_or_free(result.x, 'p_X')),
        }
        return optimized_params

    @classmethod
    def from_channel_params_optimize(cls, transmittance, N, Y_0, e_mis, p_a=None, f_EC=1.22, eps_sec=1e-10, eps_cor=1e-10, max_phase_error=0.1, initial_guess_tries=100000, mu_1=None, mu_2=None, mu_3=0, p_1=None, p_2=None, p_X=None, seed=0, debug=False):
        params = dict(transmittance=transmittance, N=N, Y_0=Y_0, e_mis=e_mis, p_a=p_a, f_EC=f_EC, eps_sec=eps_sec, eps_cor=eps_cor, max_phase_error=max_phase_error)
        fixed_params = dict(mu_1=mu_1, mu_2=mu_2, mu_3=mu_3, p_1=p_1, p_2=p_2, p_X=p_X)
        optimized_params = DecoyStateFinite.optimize_params(**params, **fixed_params, initial_guess_tries=initial_guess_tries, seed=0, debug=False)
        if debug:
            print(optimized_params)
        return cls.from_channel_params(**optimized_params, **params)

    def _vacuum_events_finite_lower_bound(self, n_mu2, n_mu3, n):
        delta = self._hoeffding_bound(n, self.eps_sec)
        n_mu3_min = max(0, n_mu3 - delta)
        n_mu2_max = n_mu2 + delta
        return vacuum_events_lower_bound(*self.mu_k, *self.p_k, n_mu2_max, n_mu3_min)

    def _single_photon_events_finite_lower_bound(self, n_mu1, n_mu2, n_mu3, s_0, n):
        """Finite-key lower bound for single-photon events."""
        delta = self._hoeffding_bound(n, self.eps_sec)
        n_mu1_max = n_mu1 + delta
        n_mu2_min = max(0, n_mu2 - delta)
        n_mu3_max = n_mu3 + delta
        return single_photon_events_lower_bound(*self.mu_k, *self.p_k, n_mu1_max, n_mu2_min, n_mu3_max, s_0)

    def _single_photon_errors_finite_upper_bound(self, m_mu2, m_mu3, m):
        """Finite-key upper bound for single-photon errors."""
        delta = self._hoeffding_bound(m, self.eps_sec)
        m_mu3_min = max(0, m_mu3 - delta)
        m_mu2_max = m_mu2 + delta
        return single_photon_errors_upper_bound(*self.mu_k, *self.p_k, m_mu2_max, m_mu3_min)

    def _single_photon_phase_error_finite_upper_bound(self, s_X1, s_Z1, nu_Z1):
        """Calculates the upper bound on the phase error rate."""
        if s_Z1 <= 0: return 1.0
        ratio = nu_Z1 / s_Z1
        g = self._gamma(self.eps_sec, ratio, s_Z1, s_X1)
        return min(ratio + g, 1.0)

    @staticmethod
    def _hoeffding_bound(n, eps, c=21):
        """Calculates the Hoeffding bound deviation."""
        return np.sqrt(n / 2 * np.log(c / eps))

    @staticmethod
    def _gamma(a, b, c, d):
        term1 = (c + d) * (1 - b) * b / (c * d * np.log(2))
        term2 = (c + d) / (c * d * (1 - b) * b) * (21**2 / a**2)
        return np.sqrt(term1 * np.log2(term2))


class DecoyStateAsymptotic():
    def __init__(self, mu_k, Q_k, E_k, Y_0=1.7e-6, f_EC=1.22):
        if any(len(arr) != 3 for arr in [mu_k, Q_k, E_k]):
            raise ValueError("Length of all array-like inputs must be 3.")
        if not all(mu >= 0 for mu in mu_k):
            raise ValueError(f'Intensities must be non-negative. {mu_k=}')
        if not mu_k[1] > mu_k[2]:
            raise ValueError(f'Constraint not met: mu_2 > mu_3. {mu_k[1]=}, {mu_k[2]=}')
        if not mu_k[0] > mu_k[1] + mu_k[2]:
            raise ValueError(f'Constraint not met: mu_1 > mu_2 + mu_3. {mu_k=}')

        self.mu_k = mu_k
        self.Q_k = np.asarray(Q_k)
        self.E_k = np.asarray(E_k)
        self.Y_0 = Y_0
        self.f_EC = f_EC

    def __repr__(self):
        mu_k = ', '.join([f"{x:.2f}" for x in self.mu_k])
        Q_k = ', '.join([f"{x:.1e}" for x in self.Q_k])
        E_k = ', '.join([f"{x:.1e}" for x in self.E_k])
        return (f"{self.__class__.__name__}("
                f"mu_k=[{mu_k}], "
                f"Q_k=[{Q_k}], "
                f"E_k=[{E_k}], "
                f"Y_0=[{self.Y_0:.2e}], "
                f"f_EC={self.f_EC:.2f}")

    def key_rate(self, clip_neg=True, debug=False):

        # Enables reuse of finite-size decoy state functions.
        # Values can be arbitrary since they cancel out in the final result.
        p_k = [1, 1, 1]

        s_0 = self.Y_0 * tau_n(*self.mu_k, *p_k, n=0)
        s_1 = single_photon_events_lower_bound(*self.mu_k, *p_k, *self.Q_k, s_0)
        Qs_1 = s_1 / tau_n(*self.mu_k, *p_k, n=1) * tau_n(*self.mu_k, *[1, 0, 0], n=1)

        EkQk = self.E_k * self.Q_k
        nu_1_upper = single_photon_errors_upper_bound(*self.mu_k, *p_k, EkQk[1], EkQk[2])
        e_ph1 = nu_1_upper / s_1 # single_photon_errors_upper_bound includes all mu_k states, so we need to divide on s_1

        if debug:
            print(f"{s_0=:.2e}, {s_1=:.2e}, {e_ph1=:.2e}")

        r = 1 / 2 * (Qs_1 * (1 - h(e_ph1)) - self.Q_k[0] * self.f_EC * h(self.E_k[0]))
        return max(r, 0) if clip_neg else r

    @classmethod
    def from_channel_params(cls, mu_k, transmittance, Y_0=1.7e-6, e_mis=0.033,
                            p_a=None, f_EC=1.22):
        params = dict(mu_k=mu_k, f_EC=f_EC)

        gains = [expected_detection_rate(transmittance, mu, Y_0) for mu in mu_k]
        qbers = [intensity_bit_error_rate(transmittance, mu, Y_0, e_mis) for mu in mu_k]

        if p_a is not None:
            qbers = [q + p_a / (2 * g) for q, g in zip(qbers, gains)]
            gains = [g * (1 + p_a) for g in gains]
        return cls(Q_k=gains, E_k=qbers, **params)

    @classmethod
    def from_channel_params_optimize(cls, transmittance, Y_0, e_mis, p_a=None, f_EC=1.22, mu_1=None, mu_2=None, mu_3=0, debug=False):
        params = dict(transmittance=transmittance, Y_0=Y_0, e_mis=e_mis, p_a=p_a, f_EC=f_EC)
        optimized_params = DecoyStateFinite.optimize_params(**params, N=1e20, p_X=1/2, debug=debug)
        return cls.from_channel_params(**params, mu_k=optimized_params['mu_k'])
