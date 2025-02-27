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
        reward = np.random.binomial(1, self.p)
        return reward

    def get_expected_value(self) -> float:
        """
        Devuelve el valor esperado de la distribución normal.

        :return: Valor esperado de la distribución.
        """
        return self.p

    def __str__(self):
        """
        Representación en cadena del brazo normal.

        :return: Descripción detallada del brazo normal.
        """
        return f"ArmBernoulli(p={self.p})"

    @classmethod
    def generate_arms(cls, k: int):
        """
        Genera k brazos con probabilidades p únicas.

        :param k: Número de brazos a generar.
        :return: Lista de brazos generados.
        """
        assert k > 0, "El número de brazos k debe ser mayor que 0."

        # Generar k- valores únicos de p con decimales
        p_values = set()
        while len(mu_values) < k:
            p = np.random().rand()
            p = round(p, 2)
            p_values.add(p)
                
        p_values = list(p_values)

        arms = [ArmBernoulli(p) for p in p_values]

        return arms
