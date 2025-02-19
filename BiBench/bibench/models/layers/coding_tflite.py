import torch
import numpy as np
import math
from sklearn.cluster import KMeans

class BinaryCodebook:
    def __init__(self, k=256,k_bits=9, max_iter=10):
        self.k = k
        self.max_iter = max_iter
        self.k_bits=k_bits

    def binarize_weights(self, tensor):
        # use where
        #return torch.sign(tensor)
        return  torch.where(tensor > 0, torch.tensor(1.0, device=tensor.device), torch.tensor(-1.0, device=tensor.device))

    def flatten_to_xbit(self, tensor):
        num_elements = tensor.numel()
        pad_size = (self.k_bits - (num_elements % self.k_bits)) % self.k_bits
        device = tensor.device
        if pad_size == 0:
            # premute here
            if tensor.dim() == 4:
                tensor = tensor.permute(0,2,3,1).contiguous()
            else:
                tensor = tensor.permute(0,2,1).contiguous()
            flattened = tensor.view(-1, self.k_bits)
            return flattened
        # premute here
        if tensor.dim() == 4:
                tensor = tensor.permute(0,2,3,1).contiguous()
        else:
            tensor = tensor.permute(0,2,1).contiguous()

        tensor = tensor.view(-1)
        tensor = torch.cat([tensor, torch.full((pad_size,), 0, dtype=tensor.dtype, device=device)])
        flattened = tensor.view(-1, self.k_bits)
        del tensor ,pad_size,num_elements
        return flattened

    def initialize_binary_codebook(self, vectors,counts):
        sorted_indices = torch.argsort(counts, descending=True)
        sorted_unique_tensor = vectors[sorted_indices]
        top_k_vectors = sorted_unique_tensor[:self.k]
        del sorted_unique_tensor ,sorted_indices
        return top_k_vectors


    def encode_vectors(self, vectors, codebook):
        # vectors = vectors.to(torch.int32)
        # codebook= codebook.to(torch.int32)
        # diff = torch.bitwise_xor(vectors.unsqueeze(1), codebook.unsqueeze(0))
        # hamming_distances = torch.sum(torch.abs(diff), dim=2)

        tensor1 = vectors.to(torch.float32)
        tensor2 = codebook.to(torch.float32)
        diff = tensor1.unsqueeze(1) - tensor2.unsqueeze(0)
        hamming_distances = torch.linalg.norm(diff, ord=1, dim=2)
        closest_indices = torch.argmin(hamming_distances, dim=1)
        closest_indices=closest_indices.to(torch.uint8)
        del vectors,codebook,diff,hamming_distances
        return closest_indices
    def process_weights(self, tensor):
        with torch.no_grad():
            # Step 1: Binarize the weights
            binarized_tensor = self.binarize_weights(tensor)

            # Step 2: Flatten the tensor into x-bit segments
            flattened_tensor = self.flatten_to_xbit(binarized_tensor)

            # Step 3: Generate or use codebook based on the number of unique vectors
            unique_vectors,counts = torch.unique(flattened_tensor, dim=0, return_counts=True)
            unique_vectors=unique_vectors
            if unique_vectors.shape[0] <= self.k:
                codebook = unique_vectors
                encoded_vectors = self.encode_vectors(flattened_tensor, codebook)
            else:
                initial_codebook = self.initialize_binary_codebook(unique_vectors,counts)
                # refined_codebook = self.refine_codebook(unique_vectors, initial_codebook)
                codebook = initial_codebook.to(torch.float32)
                del initial_codebook
                encoded_vectors = self.encode_vectors(flattened_tensor, codebook)

            # Return the codebook and encoded vectors
            del unique_vectors,flattened_tensor,binarized_tensor
            return codebook, encoded_vectors


class CodebookReplacer:
 
    @staticmethod
    def replace_with_codebook( tensor1, tensor2,coder):
         with torch.no_grad():
            # Flatten the input tensor
            flattened_tensor = coder.flatten_to_xbit(tensor1)  # Assuming the tensor is made up of x-bit vectors

            # Initialize the output tensor
            # flattened_tensor = flattened_tensor.to(torch.int32)
            # tensor2 = tensor2.to(torch.int32)
            # diff = torch.bitwise_xor(flattened_tensor.unsqueeze(1), tensor2.unsqueeze(0))
            # hamming_distances = torch.sum(torch.abs(diff), dim=2)

            flattened_tensor = flattened_tensor.to(torch.float32)
            tensor2 = tensor2.to(torch.float32)
            diff = flattened_tensor.unsqueeze(1) - tensor2.unsqueeze(0)
            hamming_distances = torch.linalg.norm(diff, ord=1, dim=2)
            # Find indices of the minimum Hamming distance for each row in tensor1
            closest_indices = torch.argmin(hamming_distances, dim=1)
            closest_indices=closest_indices.to( torch.uint8)
            del tensor1 ,tensor2, hamming_distances,diff,flattened_tensor
            return closest_indices
        
    @staticmethod
    def weight_builder(codebook,encoded_vectors,shape):
        with torch.no_grad():
            encoded_vectors = encoded_vectors.to(torch.long)
            weight_tensor = codebook[encoded_vectors]
            weight_tensor = weight_tensor.view(-1)
            num_elements= math.prod(shape)
            weight_tensor=weight_tensor[:num_elements]
            #change shape here
            if len(shape)==4:
                reordered_shape = (shape[0], shape[2], shape[3], shape[1])
            else:
                reordered_shape = (shape[0], shape[2], shape[1])

            weight_tensor = weight_tensor.view((reordered_shape))
            #premute here
            if weight_tensor.dim() == 4:
                weight_tensor = weight_tensor.permute(0,3,1,2).contiguous()
            else:
                weight_tensor = weight_tensor.permute(0,2,1).contiguous()

            del num_elements, encoded_vectors, codebook , reordered_shape
            return weight_tensor



