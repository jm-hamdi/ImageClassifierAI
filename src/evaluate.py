import tensorflow as tf
from data_loader import load_dataset
from preprocess import preprocess_data

def evaluate(model_path="../models/cnn_model.h5", dataset="mnist"):
    # Charger le modèle
    model = tf.keras.models.load_model(model_path)

    # Charger les données
    (x_train, y_train), (x_test, y_test) = load_dataset(dataset)
    x_train, x_test = preprocess_data(x_train, x_test)

    # Évaluer le modèle
    loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
    print(f"Accuracy sur test : {accuracy:.4f}")

# Cette ligne est importante pour exécuter la fonction
if __name__ == "__main__":
    evaluate()
