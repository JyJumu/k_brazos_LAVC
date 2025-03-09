"""
Module: algorithms/__init__.py
Description: Contiene las importaciones y modulos/clases públicas del paquete arms.

Authors: Jesús Verdú Chacón & Jorge López Abad
Email: jesus.v.c@um.es & jorge.lopeza@um.es
Date: 2025/02/20

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

# Importación de módulos o clases
from .src.algorithm import Algorithm
from .src.epsilon_greedy import EpsilonGreedy
from .src.ucb1 import UCB1
from .src.ucb2 import UCB2
from .src.softmax import Softmax
from .src.gradiente import Gradiente

# Lista de módulos o clases públicas
__all__ = ['Algorithm', 'EpsilonGreedy', 'UCB1', 'UCB2', 'Softmax', 'Gradiente']
