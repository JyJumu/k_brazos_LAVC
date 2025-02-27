"""
Module: arms/armbinomial.py
Description: Contains the implementation of the ArmNormal class for the normal distribution arm.

Author: Jesús Verdú Chacón
        Jorge López Abad
Email: jesus.v.c@um.es
       jorge.lopeza@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""


import numpy as np

from arms import Arm


class ArmBinomial(Arm):
    def __init__(self, n: int, p: float):
        """
        Inicializa el brazo con distribución binomial.

        :param n: Número de repeticiones del experimento
        :param p: Probabilidad de acierto.
        :param sigma: Desviación estándar de la distribución.
        """
        assert 0 <= n, "El valor n debe ser mayor o igual que cero."
        assert 0 <= p <= 1, "La probabilidad p debe estar en el intervalo [0,1]."
        
        self.n = n
        self.p = p

    def pull(self):
        """
        Genera una recompensa siguiendo una distribución binomial.

        :return: Recompensa obtenida del brazo.
        """
        reward = np.random.binomial(self.n, self.p)
        return reward

    def get_expected_value(self) -> float:
        """
        Devuelve el valor esperado de la distribución normal.

        :return: Valor esperado de la distribución.
        """
        return self.n*self.p

    def __str__(self):
        """
        Representación en cadena del brazo normal.

        :return: Descripción detallada del brazo normal.
        """
        return f"ArmBinomial(n={self.n}, p={self.p})"

    @classmethod
    def generate_arms(cls, k: int, n_min: int = 1, n_max: int = 10):
        """
        Genera k brazos con probabilidades p únicas.

        :param k: Número de brazos a generar.
        :param n_min: Número mínimo de experimentos.
        :param n_max: Número máximo de experimentos.
        :return: Lista de brazos generados.
        """
        assert k > 0, "El número de brazos k debe ser mayor que 0."
        assert n_min < n_max, "El valor de n_min debe ser menor que n_max."
        
        # Generar k- valores únicos de p con decimales
        arms = []
        for _ in range(k):
            n = np.random.randint(n_min, n_max + 1)
            n = round(n, 2)
            p = np.random().rand()
            p = round(p, 2)
            arms.append(ArmBinomial(n,p))

        return arms
