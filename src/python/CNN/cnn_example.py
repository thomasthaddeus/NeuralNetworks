"""Convolouted Neural Network Example"""

from cnn_model import CNNModel

# Example usage
cnn = CNNModel()
cnn.build_model(input_shape=(28, 28, 1), num_classes=10)
x_tr, y_tr, x_tst, y_tst = cnn.load_data()
cnn.train(x_tr, y_tr, x_tst, y_tst, epochs=10, batch_size=128)
cnn.evaluate(x_tst, y_tst)
