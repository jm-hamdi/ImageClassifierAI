# **Classification d’images avec CNN**

##  Description

Ce projet implémente un **réseau de neurones convolutif (CNN)** pour la **reconnaissance de chiffres manuscrits (MNIST)**.
L’objectif est de comprendre les CNN, l’overfitting et l’augmentation de données dans un projet pratique.

Le projet est réalisé en **Python** avec **TensorFlow / Keras**.

---

## 🛠️ Technologies utilisées

* Python 3.11
* TensorFlow 2.16 / Keras
* NumPy
* Matplotlib

---

##  Structure du projet

```
classification-cnn/
│── data/                 # MNIST téléchargé automatiquement
│── src/
│    ├── config.py        # paramètres du projet
│    ├── data_loader.py   # chargement du dataset
│    ├── preprocess.py    # preprocessing des images
│    ├── model.py         # définition du CNN
│    ├── train.py         # script d'entraînement
│    └── evaluate.py      # script d'évaluation
│── models/               # modèle sauvegardé cnn_model.h5
│── requirements.txt
│── README.md
```

---

##  Installation

1. Cloner le dépôt :

```bash
git clone https://github.com/jm-hamdi/ImageClassifierAI
cd ImageClassifierAI
```

2. Créer un environnement virtuel et l’activer :

```bash
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows
```

3. Installer les dépendances :

```bash
pip install -r requirements.txt
```

---

##  Exécution

### 1️ Entraîner le modèle

```bash
cd src
python train.py
```

* Le modèle sera entraîné sur MNIST.
* Les poids seront sauvegardés dans `../models/cnn_model.h5`.

### 2️ Évaluer le modèle

```bash
python evaluate.py
```

* Affiche la précision et la perte sur le dataset de test.

---

##  Résultats attendus

* Précision sur test MNIST : **≈ 99%**
* Perte sur test MNIST : **≈ 0.03**

---

##  Fonctionnalités avancées

* **Data Augmentation** : rotation, zoom, translation pour réduire l’overfitting.
* **Visualisation** : affichage des images et des labels pour mieux comprendre les données.

---

##  Concepts clés appris

* Réseaux de neurones convolutifs (CNN)
* Prétraitement des images et normalisation
* Overfitting et techniques pour le réduire (data augmentation, dropout)
* Entraînement et évaluation d’un modèle avec TensorFlow/Keras

---

##  Visualisation des données (optionnel)

```python
import matplotlib.pyplot as plt
from data_loader import load_dataset
from preprocess import preprocess_data

(x_train, y_train), _ = load_dataset("mnist")
x_train, _ = preprocess_data(x_train, x_train)

plt.figure(figsize=(5,5))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(x_train[i].reshape(28,28), cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
plt.show()
```

---

##  Conclusion

Ce projet fournit une base solide pour :

* Comprendre les CNN
* Travailler avec des images pour la classification
* Appliquer des techniques d’augmentation de données et de régularisation
  

---

