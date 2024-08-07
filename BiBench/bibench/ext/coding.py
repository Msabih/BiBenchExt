import torch
import numpy as np
from sklearn.cluster import KMeans

class BinaryCodebook:
    def __init__(self, k=256, max_iter=10):
        self.k = k
        self.max_iter = max_iter

    def binarize_weights(self, tensor):
        return torch.sign(tensor)

    def flatten_to_12bit(self, tensor):
        flattened = tensor.view(-1, 12)
        return flattened

    def initialize_binary_codebook(self, vectors):
        kmeans = KMeans(n_clusters=self.k, random_state=0).fit(vectors)
        initial_codebook = np.sign(kmeans.cluster_centers_)
        return initial_codebook

    def refine_codebook(self, vectors, codebook):
        for _ in range(self.max_iter):
            distances = np.linalg.norm(vectors[:, None] - codebook, axis=2)
            labels = np.argmin(distances, axis=1)
            for i in range(len(codebook)):
                if np.any(labels == i):
                    codebook[i] = np.sign(vectors[labels == i].mean(axis=0))
        return codebook

    def encode_vectors(self, vectors, codebook):
        encoded_vectors = []
        for vec in vectors:
            distances = np.linalg.norm(codebook - vec, axis=1)
            closest_index = np.argmin(distances)
            encoded_vectors.append(closest_index)
        return torch.tensor(encoded_vectors, dtype=torch.uint8)

    def process_weights(self, tensor):
        # Step 1: Binarize the weights
        binarized_tensor = self.binarize_weights(tensor)

        # Step 2: Flatten the tensor into 12-bit segments
        flattened_tensor = self.flatten_to_12bit(binarized_tensor)

        # Step 3: Generate or use codebook based on the number of unique vectors
        unique_vectors = torch.unique(flattened_tensor, dim=0).numpy()
        if unique_vectors.shape[0] <= 256:
            codebook = unique_vectors
            encoded_vectors = self.encode_vectors(flattened_tensor.numpy(), codebook)
        else:
            initial_codebook = self.initialize_binary_codebook(unique_vectors)
            refined_codebook = self.refine_codebook(unique_vectors, initial_codebook)
            codebook = torch.tensor(refined_codebook, dtype=torch.float32)
            encoded_vectors = self.encode_vectors(flattened_tensor.numpy(), codebook.numpy())

        # Return the codebook and encoded vectors
        return codebook, encoded_vectors


class CodebookReplacer:
    @staticmethod
    def hamming_distance(a, b):
        return (a ^ b).sum(dim=-1)

    def replace_with_codebook(self, tensor, codebook):
        # Flatten the input tensor
        flattened_tensor = tensor.view(-1, 12)  # Assuming the tensor is made up of 12-bit vectors

        # Initialize the output tensor
        output_tensor = torch.empty_like(flattened_tensor)

        for i, vec in enumerate(flattened_tensor):
            # Calculate the Hamming distances between the vector and all codebook entries
            distances = self.hamming_distance(vec, codebook)
            min_distance = distances.min()

            # Find indices of codebook vectors with the smallest distance
            closest_indices = torch.nonzero(distances == min_distance).squeeze()

            # If there are multiple closest vectors, randomly select one
            if len(closest_indices) > 1:
                chosen_index = closest_indices[torch.randint(0, len(closest_indices), (1,)).item()]
            else:
                chosen_index = closest_indices.item()

            # Replace the vector in the output tensor
            output_tensor[i] = codebook[chosen_index]

        # Reshape the output tensor to the original shape of the input tensor
        output_tensor = output_tensor.view(tensor.shape)
        return output_tensor

