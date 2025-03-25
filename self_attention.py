import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def self_attention(Q, K, V, dk):
    # Step 1: Q x K^T
    QK_t = np.dot(Q, K.T)
    
    # Step 2: Scale by sqrt(dk)
    scaled_QK_t = QK_t / np.sqrt(dk)
 
    # Step 3: Apply softmax
    softmax_matrix = softmax(scaled_QK_t)
    
    # Step 4: Multiply by V
    weighted_values = np.dot(softmax_matrix, V)
    
    return weighted_values

# Example matrices
Q = np.random.rand(5, 5)
K = np.random.rand(5, 5)
V = np.random.rand(5, 5)
dk = 512

# Calculate self-attention
for _ in range(100):
    result = self_attention(Q, K, V, dk)
print(result)
