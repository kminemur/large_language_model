import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

class SelfAttentionWithKVCache:
    def __init__(self, dk):
        self.dk = dk
        self.cache = {'K': None, 'V': None}

    def self_attention(self, Q, K, V):
        # Update cache
        if self.cache['K'] is None:
            self.cache['K'] = K
            self.cache['V'] = V
        else:
            self.cache['K'] = np.concatenate((self.cache['K'], K), axis=0)
            self.cache['V'] = np.concatenate((self.cache['V'], V), axis=0)
        
        # Use cached keys and values
        K = self.cache['K']
        V = self.cache['V']
        
        # Step 1: Q x K^T
        QK_t = np.dot(Q, K.T)
        
        # Step 2: Scale by sqrt(dk)
        scaled_QK_t = QK_t / np.sqrt(self.dk)
     
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

# Create SelfAttentionWithKVCache instance
self_attention_with_cache = SelfAttentionWithKVCache(dk)


# Calculate self-attention with KV cache for 100 iterations
for _ in range(100):
    result = self_attention_with_cache.self_attention(Q, K, V)
print(result)
