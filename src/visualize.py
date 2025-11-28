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
