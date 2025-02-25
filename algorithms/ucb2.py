"""
Module: algorithms/ucb2.py
Description: Implementación del algoritmo ucb2 para el problema de los k-brazos.

Author: Jesús Verdú Chacón & Jorge López Abad
Email: jesus.v.c@um.es & jorge.lopeza@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

import numpy as np
import math
from algorithms.algorithm import Algorithm

class UCB2(Algorithm):

    def __init__(self, k: int, alfa: float = 0.1):
        """
        Inicializa el algoritmo UCB.

        :param k: Número de brazos.
        :param alfa: Parámetro de ajuste entre exploración y explotación.
        :raises ValueError: Si epsilon no está en [0, 1].
        """
        assert 0 < alfa < 1, "El parámetro alfa debe estar entre 0 y 1."

        super().__init__(k)
        self.alfa = alfa
        self.uas: np.ndarray = np.zeros(k, dtype=float)
        self.ucbs: np.ndarray = np.zeros(k, dtype=float)
        self.kas: np.darray = np.zeros(k, dtype=int)

    def tau(self, ka: int) -> int:
        return math.ceil((1 + self.alfa)**ka)

    def select_arm(self, t: int) -> int:
        """
        Selecciona un brazo basado en la política UCB1.
        :param t: instante de tiempo en el que nos encontramos
        :return: índice del brazo seleccionado.
        """

        for i in range(self.k):
            self.uas[i] = np.sqrt((1 + self.alfa) * np.log(math.e * t / self.tau(self.kas[i])) / 2 * self.tau(self.kas[i]))
        
        for i in range(self.k):
            self.ucbs[i] = self.values[i] + self.uas[i]

        chosen_arm = np.argmax(self.ucbs)

        return chosen_arm

    def update(self, chosen_arm: int, reward: float):
        num_veces = self.tau(self.kas[chosen_arm] + 1) - self.tau(self.kas[chosen_arm])
        for i in range(num_veces):
            super().update(chosen_arm, reward)

        self.taus[chosen_arm] = 1 + (1 + self.alfa)**self.kas[chosen_arm]
        self.kas[chosen_arm] += 1
        
