"""
Module: algorithms/ucb1.py
Description: Implementación del algoritmo ucb1 para el problema de los k-brazos.

Author: Jesús Verdú Chacón & Jorge López Abad
Email: jesus.v.c@um.es & jorge.lopeza@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

import numpy as np

from algorithms.algorithm import Algorithm

class UCB1(Algorithm):

    def __init__(self, k: int, c: float = 1):
        """
        Inicializa el algoritmo UCB.

        :param k: Número de brazos.
        :param c: Parámetro de ajuste de exploración.
        :raises ValueError: Si c no está en [0, 1].
        """
        assert 0 <= c <= 1, "El parámetro c debe estar entre 0 y 1."

        super().__init__(k)
        self.c = c
        self.uas: np.ndarray = np.zeros(k, dtype=float)
        self.ucbs: np.ndarray = np.zeros(k, dtype=float)

    def select_arm(self, t: int) -> int:
        """
        Selecciona un brazo basado en la política UCB1.
        :param t: instante de tiempo en el que nos encontramos
        :return: índice del brazo seleccionado.
        """

        # Primero seleccionamos todos los brazos para tener las recompensas
        for i in range(self.k):
            if self.counts[i] == 0:
                return i
        
        for i in range(self.k):
            self.uas[i] = np.sqrt(2 * np.log(t+1) / self.counts[i])
        
        for i in range(self.k):
            self.ucbs[i] = self.values[i] + self.c * self.uas[i]

        chosen_arm = np.argmax(self.ucbs)

        return chosen_arm

    def reset(self):
        """
        Reinicia el estado del algoritmo (opcional).
        """
        self.counts = np.zeros(self.k, dtype=int)
        self.values = np.zeros(self.k, dtype=float)
        self.uas = np.zeros(self.k, dtype=float)
        self.ucbs = np.zeros(self.k, dtype=float)
