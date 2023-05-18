"""2. RNN for Sequence Data Processing

The second part of your task could involve using an RNN to process sequence data. Here's an example of how you might construct such a network:"""

from tensorflow.keras.layers import Embedding, LSTM, Dense

# Add an embedding layer
model.add(Embedding(input_dim=10000, output_dim=64))

# Add LSTM layers
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128))

# Add a Dense layer
model.add(Dense(1, activation='sigmoid'))
