# IA Aplicada al Marketing Digital - Proyecto 1

Este es el primer proyecto de la asignatura de Inteligencia Artificial Aplicada (AIA), enfocado en optimizar estrategias de Marketing Digital mediante el uso de modelos de aprendizaje automático y análisis de datos.

## 🚀 Descripción del Proyecto
El sistema desarrollado automatiza un flujo completo para los datos: desde la ingesta hasta la evaluación de modelos y la inferencia personalizada, permitiendo que las empresas puedan indentificar qué usurios tienen mayor probabilidad de compra.

### Objetivos principales:

* **Análisis y Procesamiento:** Limpieza de datos, mapeo temporal y gestión del desequilibrio de clases.
* **Predicción de Conversión:** Desarrollo de modelos de clasificación para prever la probabilidad de compra.
* **Sistema de inferencia flexible**: Capacidad de usar los modelos entrenado con usuarios predefinidos y cargar nuevos perfiles mediante archivos JSON.

## 🛠️ Tecnologías Utilizadas

* **Lenguaje:** Python 3.12+
* **Librerías principales:**
    * **Modelado**: `Scikit-learn` (Random Forest) y `XGBoost`.
    * **Procesamiento**: `Pandas`, `Numpy`, e `Imbalanced-learn` (SMOTE).
    * **Visualización**: `Matplotlib` y `Seaborn` 

## 📁 Estructura del Repositorio

* `notebooks/`: EDA detallado con visualización y experimentos realizados.
* `src/`: Código fuente organizado de manera modular
  * `main.py`: Script principal de ejecución
  * `utils/data.py`: Carga, preparación de datos y gestión de JSONs.
  * `utils/models.py`: Entrenamiento y lógica de predicción.
  * `utils/interfaz.py`: Funciones para la navegación por menús en terminal
* `users/`: Directorio destinado a almacenar los archivos `.json` con perfiles de usuario para inferencia personalizadas.

## ⚙️ Instalación y Uso

1.  **Crear un entorno virtual (recomendado):**
    ```bash
    python -m venv .venv
    source venv/bin/activate  # En Windows: .venv\Scripts\activate
    ```

2.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```
## 🎛️ Guia de Uso del Script Principal
Para iniciar el sistema se debe de ejecutar:

```bash
python src/main.py
```

Seguidamente el script te guiará mediante consola a través de los siguientes pasos:

1. **Preparación y Resumen**. El sistema carga los datos automáticamente, aplicará las transformaciones necesarias y mostrará por pantalla un resumen del conjunto de entrenamiento.
2. **Selección de Modelo**. Deberás elegir mediante teclado que algoritmo se desea entrenar:
   1. RandomForest
   2. XGBoost
3. **Fase de inferencia**. Podrás elegir entre dos métodos para probar el modelo:
   1. **Usuario por defecto**. Utiliza un perfil preconfigurado en el código para una prueba rápida.
   2. Carga desde JSON. Permite introducir el nombre de un archivo (ej: `example.json`). **IMPORTANTE**, los archivos deben de estar ubicados dentro del directorio `users/`.
---
## Contributors

[![contributors](https://contrib.rocks/image?repo=alvaroG-IA/proyecto-1-AIA-Marketing_Digital)](https://github.com/alvaroG-IA/proyecto-1-AIA-Marketing_Digital/graphs/contributors)
