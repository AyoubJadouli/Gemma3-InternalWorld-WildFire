import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

class Gemma3InternalWorldModel(nn.Module):
    """
    Multi-branch neural network with a frozen Gemma 3 internal world layer
    for wildfire prediction.
    
    Architecture:
    1. Input features split into 4 branches
    2. Each branch processed through its own FFN
    3. Branches concatenated to form a 1152-dimensional vector
    4. 3-layer FFN processing
    5. Projection to final 1152 dims
    6. Frozen Gemma3 partial decoder layer
    7. Final classification MLP
    """
    def __init__(self, n_features, gemma_path="google/gemma-3-1b-it", dropout_rate=0.3):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = 1152  # Gemma 3-1B hidden dimension
        self.dropout_rate = dropout_rate
        
        # Split features into 4 branches (approximately equal size)
        self.branch_size = n_features // 4
        self.branch_sizes = [self.branch_size] * 3
        self.branch_sizes.append(n_features - sum(self.branch_sizes[:3]))
        
        # Each branch outputs 1152/4 = 288 dimensions
        self.branch_output_dim = self.hidden_dim // 4
        
        # Define the 4 parallel branches
        self.branch1 = self._make_branch(self.branch_sizes[0], self.branch_output_dim)
        self.branch2 = self._make_branch(self.branch_sizes[1], self.branch_output_dim)
        self.branch3 = self._make_branch(self.branch_sizes[2], self.branch_output_dim)
        self.branch4 = self._make_branch(self.branch_sizes[3], self.branch_output_dim)
        
        # 3-layer FFN after concatenation
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim)
        )
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim)
        )
        
        # Load Gemma 3 decoder layer
        self.gemma_layer = self._load_gemma_layer(gemma_path)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 1)
        )
        
        # Initialize weights for better training
        self._initialize_weights()
    
    def _make_branch(self, in_dim, out_dim):
        """Create a feedforward branch with ReLU activation."""
        return nn.Sequential(
            nn.Linear(in_dim, out_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(out_dim // 2),
            nn.Linear(out_dim // 2, out_dim),
            nn.ReLU(),
            nn.LayerNorm(out_dim)
        )
    
    def _load_gemma_layer(self, gemma_path):
        """
        Load a Gemma 3 decoder layer - either real (if transformers available) 
        or a simulated replacement.
        """
        # Try to import the transformers library
        try:
            from transformers import Gemma3ForCausalLM
            HAS_TRANSFORMERS = True
            print("Transformers library is available.")
        except ImportError:
            HAS_TRANSFORMERS = False
            print("Transformers library is not available. Will use simulated layers.")
        
        # Simulated Gemma layer for testing or when transformers not available
        class SimulatedGemmaLayer(nn.Module):
            def __init__(self, hidden_size=1152):
                super().__init__()
                self.attn = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
                self.ln1 = nn.LayerNorm(hidden_size)
                self.ln2 = nn.LayerNorm(hidden_size)
                self.ffn = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 4),
                    nn.GELU(),
                    nn.Linear(hidden_size * 4, hidden_size)
                )
                
            def forward(self, hidden_states, **kwargs):
                # Simple transformer decoder layer logic
                residual = hidden_states
                hidden_states = self.ln1(hidden_states)
                # Self-attention (we know seq_len=1 so no mask needed)
                attn_output, _ = self.attn(hidden_states, hidden_states, hidden_states)
                hidden_states = residual + attn_output
                
                # FFN
                residual = hidden_states
                hidden_states = self.ln2(hidden_states)
                hidden_states = self.ffn(hidden_states)
                hidden_states = residual + hidden_states
                
                return hidden_states
        
        # If transformers not available or loading fails, use simulated layer
        if not HAS_TRANSFORMERS:
            print("Using simulated Gemma decoder layer")
            return SimulatedGemmaLayer()
        
        try:
            print(f"Attempting to load Gemma model from {gemma_path}...")
            gemma_model = Gemma3ForCausalLM.from_pretrained(
                gemma_path,
                torch_dtype=torch.float16,
                device_map=None,
                local_files_only=False
            )
            
            # Disable sliding window if present in config
            if hasattr(gemma_model.config, "is_sliding"):
                gemma_model.config.is_sliding = False
            
            # Freeze all parameters in the Gemma model
            for param in gemma_model.parameters():
                param.requires_grad = False
            
            # Extract a single decoder layer (middle of the model)
            all_layers = gemma_model.model.layers
            n_layers = len(all_layers)
            layer_idx = n_layers // 2  # Middle layer
            gemma_layer = all_layers[layer_idx]
            
            print(f"Successfully loaded Gemma decoder layer {layer_idx} of {n_layers}")
            return gemma_layer
            
        except Exception as e:
            print(f"Error loading Gemma model: {e}")
            print("Falling back to simulated layer.")
            return SimulatedGemmaLayer()
    
    def _initialize_weights(self):
        """Initialize the weights of the trainable layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Only apply xavier to weight matrices (2+ dims), not biases
                if m.weight.dim() > 1:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass through the model.
        Input: x - batch of feature vectors (B, n_features)
        Output: Wildfire probability (B, 1)
        """
        # Split input into 4 branches
        x1 = x[:, :self.branch_sizes[0]]
        x2 = x[:, self.branch_sizes[0]:self.branch_sizes[0]+self.branch_sizes[1]]
        x3 = x[:, self.branch_sizes[0]+self.branch_sizes[1]:sum(self.branch_sizes[:3])]
        x4 = x[:, sum(self.branch_sizes[:3]):]
        
        # Process each branch
        b1 = self.branch1(x1)
        b2 = self.branch2(x2)
        b3 = self.branch3(x3)
        b4 = self.branch4(x4)
        
        # Concatenate branch outputs to form 1152-dim vector
        concatenated = torch.cat([b1, b2, b3, b4], dim=1)  # (B, 1152)
        
        # Apply 3-layer FFN
        ffn_output = self.ffn(concatenated)
        
        # Apply projection
        projection = self.projection(ffn_output)  # (B, 1152)
        
        # Reshape for Gemma layer: (B, 1152) -> (B, 1, 1152)
        gemma_input = projection.unsqueeze(1)
        
        # Cast to float16 if the Gemma layer is in float16
        if hasattr(self.gemma_layer, 'self_attn') and self.gemma_layer.self_attn.q_proj.weight.dtype == torch.float16:
            gemma_input = gemma_input.half()

        # Forward pass through Gemma layer
        gemma_output = self.gemma_layer(gemma_input)
        
        # Cast back to float32 if needed
        if gemma_output.dtype != torch.float32:
            gemma_output = gemma_output.float()
        
        # Squeeze sequence dimension: (B, 1, 1152) -> (B, 1152)
        gemma_output = gemma_output.squeeze(1)
        
        # Classification head
        output = self.classifier(gemma_output)
        
        # Apply sigmoid for probability output
        return torch.sigmoid(output)

    def count_parameters(self):
        """Count total and trainable parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': total_params - trainable_params,
            'trainable_percentage': (trainable_params / total_params) * 100
        }

# Example usage
if __name__ == "__main__":
    # Create a model instance with 276 input features
    model = Gemma3InternalWorldModel(n_features=276)
    
    # Count parameters
    params = model.count_parameters()
    print(f"Model has {params['trainable']:,} trainable parameters out of {params['total']:,} total")
    print(f"Percentage of trainable parameters: {params['trainable_percentage']:.2f}%")
    
    # Generate a random input tensor
    batch_size = 16
    n_features = 276
    x = torch.randn(batch_size, n_features)
    
    # Forward pass
    output = model(x)
    print(f"Output shape: {output.shape}")  # Should be [16, 1]
