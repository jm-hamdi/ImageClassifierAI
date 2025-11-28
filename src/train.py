from data_loader import load_dataset
from preprocess import preprocess_data
from model import create_cnn_model
from config import EPOCHS

def train_model(dataset="mnist"):
    (x_train, y_train), (x_test, y_test) = load_dataset(dataset)
    x_train, x_test = preprocess_data(x_train, x_test)

    model = create_cnn_model()
    history = model.fit(x_train, y_train, epochs=EPOCHS, validation_split=0.1)

    model.save("../models/cnn_model.h5")
    return history

if __name__ == "__main__":
    train_model("mnist")
