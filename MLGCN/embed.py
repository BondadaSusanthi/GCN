import numpy as np
import pickle

# Labels order in CSV (ODIR standard)
labels = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
num_classes = len(labels)

# Create Identity Matrix
embedding_matrix = np.eye(num_classes)

# Save to label_embeddings.pkl
with open('embeddings.pkl', 'wb') as f:
    pickle.dump(embedding_matrix, f)

print(f"label_embeddings.pkl created with shape {embedding_matrix.shape}")
