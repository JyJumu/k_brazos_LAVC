"""
Module: arms/armbernoulli.py
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


class ArmBernoulli(Arm):
    def __init__(self, p: float):
        """
        Inicializa el brazo con distribución bernoulli.

        :param p: Probabilidad de acierto.
        :param sigma: Desviación estándar de la distribución.
        """
        assert 0 <= p <= 1, "La probabilidad p debe estar en el intervalo [0,1]."

        self.p = p

    def pull(self):
        """
        Genera una recompensa siguiendo una distribución normal.

        :return: Recompensa obtenida del brazo.
        """
        reward = np.random.normal(self.mu, self.sigma)
        return reward

    def get_expected_value(self) -> float:
        """
        Devuelve el valor esperado de la distribución normal.

        :return: Valor esperado de la distribución.
        """
        prob = np.random().rand()
        if prob < p:
          return 1
        else:
          return 0

    def __str__(self):
        """
        Representación en cadena del brazo normal.

        :return: Descripción detallada del brazo normal.
        """
        return f"ArmNormal(mu={self.mu}, sigma={self.sigma})"

    @classmethod
    def generate_arms(cls, k: int, mu_min: float = 1, mu_max: float = 10.0):
        """
        Genera k brazos con medias únicas en el rango [mu_min, mu_max].

        :param k: Número de brazos a generar.
        :param mu_min: Valor mínimo de la media.
        :param mu_max: Valor máximo de la media.
        :return: Lista de brazos generados.
        """
        assert k > 0, "El número de brazos k debe ser mayor que 0."
        assert mu_min < mu_max, "El valor de mu_min debe ser menor que mu_max."

        # Generar k- valores únicos de mu con decimales
        mu_values = set()
        while len(mu_values) < k:
            mu = np.random.uniform(mu_min, mu_max)
            mu = round(mu, 2)
            mu_values.add(mu)

        mu_values = list(mu_values)
        sigma = 1.0

        arms = [ArmNormal(mu, sigma) for mu in mu_values]

        return arms
