from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os


class MetaPopModel(object):
    """
    Class for Metapopulation model
    """

    def __init__(
        self,
        net_params: np.ndarray, # A 3x3 matrix of alpha_ij
        beta: float,
        gamma: float,
        pop_size: np.ndarray,
    ) -> None:
        self.net_params = net_params
        self.beta = beta
        self.gamma = gamma
        self.pop_size = pop_size
        self.communities = pop_size.shape[0]

    def ode(self, times, init, parms):
        """Metapopulation model ODE"""

        # reshapes init array into a matrix with the rows being 
        # the number of communities and the columns being 3 (S, I, R)
        init_ = init.reshape((self.communities, 3))

        self.pop_size = init_.sum(axis=1)
        S, I, R = init_[:, 0], init_[:, 1], init_[:, 2]
        # S,I,R are now 1D arrays of length self.communities
        # S, I, R represent the number of susceptible, infected
        # and recovered individuals in each community
        beta, gamma = parms

        S_eff = None
        I_eff = None
        R_eff = None
        ############################################################
        # Compute effective S, I, R from S, I, R, pop_size and net_params
        # YOUR CODE HERE
        
        n1 = self.pop_size[0]
        n2 = self.pop_size[1]
        n3 = self.pop_size[2]
        
        sig12 = self.net_params[0, 1]
        sig13 = self.net_params[0, 2]
        sig21 = self.net_params[1, 0]
        sig23 = self.net_params[1, 2]
        sig31 = self.net_params[2, 0]
        sig32 = self.net_params[2, 1]

        # S 
        S1 = S[0]
        S2 = S[1]
        S3 = S[2]
        
        s1_in1 = (S2 * (sig21 / n2)) + (S3 * (sig31 / n3))
        s1_out1 = (S1 * (sig12 / n1)) + (S1 * (sig13 / n1))
        S_eff1 = S1 + (s1_in1 - s1_out1)
        
        s2_in2 = (S1 * (sig12 / n1)) + (S3 * (sig32 / n3))
        s2_out2 = (S2 * (sig21 / n2)) + (S2 * (sig23 / n2))
        S_eff2 = S2 + (s2_in2 - s2_out2)
        
        s3_in3 = (S1 * (sig13 / n1)) + (S2 * (sig23 / n2))
        s3_out3 = (S3 * (sig31 / n3)) + (S3 * (sig32 / n3))
        S_eff3 = S3 + (s3_in3 - s3_out3)
        
        # I
        I1 = I[0]
        I2 = I[1]
        I3 = I[2]
        
        i1_in1 = (I2 * (sig21 / n2)) + (I3 * (sig31 / n3))
        i1_out1 = (I1 * (sig12 / n1)) + (I1 * (sig13 / n1))
        I_eff1 = I1 + (i1_in1 - i1_out1)
        
        i2_in2 = (I1 * (sig12 / n1)) + (I3 * (sig32 / n3))
        i2_out2 = (I2 * (sig21 / n2)) + (I2 * (sig23 / n2))
        I_eff2 = I2 + (i2_in2 - i2_out2)
        
        i3_in3 = (I1 * (sig13 / n1)) + (I2 * (sig23 / n2))
        i3_out3 = (I3 * (sig31 / n3)) + (I3 * (sig32 / n3))
        I_eff3 = I3 + (i3_in3 - i3_out3)
        
        S_eff = np.array([S_eff1, S_eff2, S_eff3])
        I_eff = np.array([I_eff1, I_eff2, I_eff3])
        
        ############################################################
    

        dSdt = np.outer(S_eff, -beta * I_eff / self.pop_size).sum(axis=1)
        dIdt = -dSdt - gamma * I_eff
        dRdt = gamma * I_eff
        
        return np.array([dSdt, dIdt, dRdt]).T.ravel()

    def solve(self, init: List[float], parms: List[float], times: np.ndarray):
        """Solve Metapopulation model"""
        sol = solve_ivp(
            lambda t, y: self.ode(t, y, parms),
            (times[0], times[-1]),
            init,
            t_eval=times,
        )
        return sol.y

    def plot_soln(
        self,
        init: List[float],
        parms: List[float],
        times: np.ndarray,
        save_path: Optional[str] = None,
    ):
        sol = self.solve(init, parms, times)
        sol = sol.reshape((self.communities, 3, -1))
        for i, s in enumerate(sol):
            save_path_pop = f"{save_path}_{i}.png" if save_path else None
            self.plot(s[0], s[1], s[2], save_path_pop)

    def plot(self, s, i, r, save_path: Optional[str] = None):
        """Plot Metapopulation model"""
        plt.clf()
        plt.plot(s, label="S")
        plt.plot(i, label="I")
        plt.plot(r, label="R")
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Population fraction")
        if save_path:
            # Make dir if no exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        plt.show()


if __name__ == "__main__":
    pops = np.array([1000.0, 200.0, 300.0])
    params = [0.5, 0.3]
    net_params = (
        np.array([[0.0, 0.51, 0.10], [0.02, 0.0, 0.10], [0.02, 0.01, 0.0]]) * pops
    )

    init = np.array([[900.0, 100.0, 0.0], [100.0, 10.0, 90.0], [100.0, 50.0, 150.0]])

    model = MetaPopModel(net_params, params[0], params[1], np.array(pops))
    model.plot_soln(init.ravel(), params, np.linspace(0, 100, 1000), "plots/metapop")
    # soln = model.solve(init.ravel(), params, np.linspace(0, 100, 1000))
