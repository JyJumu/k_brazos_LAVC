"""
Module: plotting/plotting.py
Description: Contiene funciones para generar gráficas de comparación de algoritmos.

Author: Jesús Verdú Chacón
        Jorge López Abad
Email: jesus.v.c@um.es
       jorge.lopeza@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

from typing import List

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from algorithms import Algorithm, EpsilonGreedy, UCB1, UCB2


def get_algorithm_label(algo: Algorithm) -> str:
    """
    Genera una etiqueta descriptiva para el algoritmo incluyendo sus parámetros.

    :param algo: Instancia de un algoritmo.
    :type algo: Algorithm
    :return: Cadena descriptiva para el algoritmo.
    :rtype: str
    """
    label = type(algo).__name__
    if isinstance(algo, EpsilonGreedy):
        label += f" (epsilon={algo.epsilon})"
    elif isinstance(algo, UCB1):
        label += f" (c={algo.c})"
    elif isinstance(algo, UCB2):
        label += f" (α={algo.alfa})"
    else:
        raise ValueError("El algoritmo debe ser de la clase Algorithm o una subclase.")
    return label


def plot_average_rewards(steps: int, rewards: np.ndarray, algorithms: List[Algorithm]):
    """
    Genera la gráfica de Recompensa Promedio vs Pasos de Tiempo.

    :param steps: Número de pasos de tiempo.
    :param rewards: Matriz de recompensas promedio.
    :param algorithms: Lista de instancias de algoritmos comparados.
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), rewards[idx], label=label, linewidth=2)

    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('Recompensa Promedio', fontsize=14)
    plt.title('Recompensa Promedio vs Pasos de Tiempo', fontsize=16)
    plt.legend(title='Algoritmos')
    plt.tight_layout()
    plt.show()

def plot_optimal_selections(steps: int, optimal_selections: np.ndarray, algorithms: List[Algorithm]):
    """
    Genera la gráfica de Porcentaje de Selección del Brazo Óptimo vs Pasos de Tiempo.

    :param steps: Número de pasos de tiempo.
    :param optimal_selections: Matriz de porcentaje de selecciones óptimas.
    :param algorithms: Lista de instancias de algoritmos comparados.
    """

    plt.figure(figsize=(10, 6))

    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), optimal_selections[idx], label=label, linewidth=2)

    plt.xlabel("Pasos de tiempo")
    plt.ylabel("Porcentaje de selecciones óptimas (%)")
    plt.title("Porcentaje de Selección del Brazo Óptimo vs Pasos de Tiempo")
    plt.legend()
    plt.grid()
    plt.show()

def plot_regret(steps: int,
regret_accumulated: np.ndarray,
algorithms: List[Algorithm], *args):
    """
    Genera la gráfica de Regret Acumulado vs Pasos de Tiempo
    :param steps: Número de pasos de tiempo.
    :param regret_accumulated: Matriz de regret acumulado (algoritmos x pasos).
    :param algorithms: Lista de instancias de algoritmos comparados.
    :param args: Opcional. Parámetros que consideres. P.e. la cota teórica Cte * ln(T).
    """

    plt.figure(figsize=(10, 6))

    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), regret_accumulated[idx], label=label, linewidth=2)
            
    plt.xlabel("Pasos de tiempo")
    plt.ylabel("Arrepentimiento acumulado")
    plt.title("Arrepentimiento acumulado vs Pasos de Tiempo")
    plt.legend()
    plt.grid()
    plt.show()

def plot_arm_statistics(arm_stats: np.ndarray,
algorithms: List[Algorithm], k, *args):
    """
    Genera una gráfica mostrando la ganancia obtenida para cada brazo.
    :param arm_stats: Lista (de diccionarios) con estadísticas de cada brazo por algoritmo.
    :param algorithms: Lista de instancias de algoritmos comparados.
    :param args: Opcional. Parámetros que consideres
    """

    plt.figure(figsize=(10, 6))
        
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(k), arm_stats[idx], label=label, linewidth=2)

    plt.xlabel("Brazos del bandido")
    plt.ylabel("Ganancia obtenida")
    plt.title("Ganancia obtenida por cada brazo")
    plt.legend()
    plt.grid()
    plt.show()

def plot_arm_num_choices(num_choices_arm: np.ndarray,
                        algorithms: List[Algorithm], k, *args):
    """
    Genera una gráfica mostrando el número de elecciones de cada brazo.
    :param num_choices_arm: Lista que contiene el número de elecciones de cada brazo
    por algoritmo.
    :param algorithms: Lista de instancias de algoritmos comparados.
    :param args: Opcional. Parámetros que consideres
    """

    plt.figure(figsize=(10, 6))
        
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(k), num_choices_arm[idx], label=label, linewidth=2)

    plt.xlabel("Brazos del bandido")
    plt.ylabel("Número de elecciones")
    plt.title("Número de elecciones por cada brazo")
    plt.legend()
    plt.grid()
    plt.show()
