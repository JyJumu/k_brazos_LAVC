"""
Module: algorithms/gradiente.py
Description: Implementación del algoritmo ucb1 para el problema de los k-brazos.

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

class Gradiente(Algorithm):

    def __init__(self, k: int, alfa: float):
        """
        Inicializa el algoritmo UCB.

        :param k: Número de brazos.
        :param alfa: Tasa de aprendizaje para actualizar las Hs.
        :raises ValueError: Si tau no es mayor que 0.
        """
        assert 0 <= alfa, "El parámetro tau debe se mayor que 0."

        super().__init__(k)
        self.alfa = alfa
        self.prob : np.ndarray = np.zeros(k, dtype=float)
        self.hs : np.ndarray = np.zeros(k, dtype=float)
        self.average_reward = 0.0

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política Softmax.
        :return: índice del brazo seleccionado.
        """

        self.probs = np.exp(self.hs) / np.sum(np.exp(self.hs))
        chosen_arm = np.random.choice(self.k, p = self.probs)
        return chosen_arm

    def update(self, chosen_arm: int, reward: float, t: int):
        """
        Actualiza las recompensas promedio estimadas de cada brazo.
        :param chosen_arm: Índice del brazo que fue tirado.
        :param reward: Recompensa obtenida.
        :param r: Paso de tiempo en el que nos encontramos.
        """

        self.average_rewards += (reward - self.average_rewards) / (t+1)
        for i in range(self.k):
            is_chosen_arm = int(i == chosen_arm)
            self.hs[i] += self.alfa * (reward - self.average_rewards) * (is_chosen_arm - self.probs[i])


        super().update(chosen_arm, reward)

    def reset(self):
        """
        Reinicia el estado del algoritmo (opcional).
        """
        self.counts = np.zeros(self.k, dtype=int)
        self.values = np.zeros(self.k, dtype=float)
        self.prob = np.zeros(self.k, dtype=float)
        self.hs = np.zeros(self.k, dtype=float)
        self.average_rewards = 0.0
