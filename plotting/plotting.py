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

from algorithms import Algorithm, EpsilonGreedy, UCB1, UCB2, Softmax, Gradiente
from typing import List, Dict


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
    elif isinstance(algo, Softmax):
        label += f" (τ={algo.tau})"
    elif isinstance(algo, Gradiente):
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

def plot_arm_statistics(arm_stats: List[Dict], algorithms: List, k: int, optimal_arm: int, *args):
    """
    Genera gráficas en un diseño de dos columnas mostrando la selección de brazos y
    la ganancia obtenida por cada brazo por algoritmo.

    :param arm_stats: Lista (de diccionarios) con estadísticas de cada brazo por algoritmo.
    :param algorithms: Lista de instancias de algoritmos comparados.
    :param k: Número de brazos.
    :param optimal_arm: Índice del brazo óptimo.
    """
    num_algorithms = len(algorithms)
    cols = 2  # Número de columnas en la cuadrícula
    rows = (num_algorithms + 1) // cols  # Calculamos las filas necesarias
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6 * rows))
    axes = axes.flatten() if num_algorithms > 1 else [axes]
    
    for idx, algo in enumerate(algorithms):
        ax = axes[idx]
        label = get_algorithm_label(algo)
        mean_rewards = arm_stats[idx]['mean_rewards']
        selections = arm_stats[idx]['selections']
        
        x_positions = np.arange(1, k + 1)
        x_labels = [f"{i}\n({int(selections[i - 1])} veces)" for i in range(1, k + 1)]
        
        colors = ['green' if (i - 1) == optimal_arm else 'blue' for i in x_positions]
        
        ax.bar(x_positions, mean_rewards, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xlabel("Brazos del bandido (Número de selecciones entre paréntesis)")
        ax.set_ylabel("Ganancia promedio")
        ax.set_title(f"Ganancia promedio por brazo - {label}")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Ocultar los ejes sobrantes si el número de algoritmos es impar
    for idx in range(num_algorithms, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
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
