# k_brazos_LAVC
## Información
- **Alumnos:** Verdú Chacón, Jesús; López Abad, Jorge
- **Asignatura:** Extensiones de Machine Learning
- **Curso:** 2024/2025
- **Grupo:** LAVC
## Descripción
En este trabajo se ha realiazdo un estudio comparativo para estudiar el rendimiento de algoritmos de las familias $\epsilon$-greedy, softmax y UCB sobre el problema de aprendizaje por refuerzo del bandido multibrazo. En concreto, de la familia $\epsilon$-greedy se ha utilizado el algoritmo $\epsilon$-greedy con diferentes valores de $\epsilon$ sobre un bandido de 10 brazos con distribución de recompensas normal; y de las familaias softmax y UCB se han utilizado los algoritmos softmax, gradiente de preferencias, UCB1 y UCB2 sobre tres bandidos de 10 brazos con distribuciones de recompensa normal, binomial y Bernoulli

## Estructura
En la carpeta "algorithms" se hallan los ficheros donde se desarrollan los algoritmos anteriormente comentados. En la carpeta "arms" se halla el código para poder utilizar bandidos multibrazo con las distribuciones de recompensa normal, binomial y Bernoulli. Y en la carpeta "plotting" se halla el código para poder dibujar las gráficas utilizadas para la realización del estudio.

En la carpeta principal se hallan todos los ficheros Jupyter Notebook donde se han realizado los experimentos. El nombre de estos ficheros sigue la estructura "[familia del algoritmo]\_EML\_[distribución de recompensa utilizada].ipynb", donde [familia del algoritmo] hace referencia a la familia del algoritmo sobre la cual hemos realizado el experimento, siendo estas _epsilongreedy_, _Softmax_, la cual incluye a los algoritmos softmax y gradiente de preferencias; y _UCB_, que incluye los algoritmos UCB1 y UCB2.

El fichero "main.ipynb" sirve como enlace para poder acceder a todos los notebooks anteriores.

## Instalación y Uso
No es necesaria realizar ninguna instalación para poder reproducir los experimentos, basta entrar en los ficheros Jupyter Notebook donde se hallan los experimentos y abrirlos en Colab. Todos los experimentos son reproducibles.
## Tecnologías Utilizadas
Hemos desarrollado el código necesario para poder llevar a cabo los experimentos en ficheros .py de lenguaje Python. Los experimentos se han desarrollado en ficheros Jupyter Notebook
