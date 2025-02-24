"""
Module: algorithms/ucb1.py
Description: Implementación del algoritmo epsilon-greedy para el problema de los k-brazos.

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

    def __init__(self, k: int, alfa: float = 0.1):
        """
        Inicializa el algoritmo UCB.

        :param k: Número de brazos.
        :param epsilon: Probabilidad de exploración (seleccionar un brazo al azar).
        :raises ValueError: Si epsilon no está en [0, 1].
        """
        assert 0 < alfa < 1, "El parámetro alfa debe estar entre 0 y 1."

        super().__init__(k)
        self.alfa = alfa
        self.uas: np.ndarray = np.zeros(k, dtype=float) 

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política UCB1.
        :return: índice del brazo seleccionado.
        """

        
        
        if np.random.random() < self.epsilon:
            # Selecciona un brazo al azar
            chosen_arm = np.random.choice(self.k)
        else:
            # Selecciona el brazo con la recompensa promedio estimada más alta
            chosen_arm = np.argmax(self.values)

        return chosen_arm
