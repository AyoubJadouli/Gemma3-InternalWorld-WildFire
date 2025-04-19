import torch
import torch.nn as nn
import torch.nn.functional as F


class FFNModel(nn.Module):
    """
    Standard Feed-Forward Neural Network for baseline comparison.
    
    A simple MLP with three hidden layers and batch normalization.
    """
    def __init__(self, n_features, dropout_rate=0.3):
        super(FFNModel, self).__init__()
        self.dense1 = nn.Linear(n_features, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.dense2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.dense3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        
        self.output_layer = nn.Linear(64, 1)
    
    def forward(self, x):
        x = F.gelu(self.dense1(x))
        x = self.bn1(x)
        x = self.dropout1(x)
        
        x = F.gelu(self.dense2(x))
        x = self.bn2(x)
        x = self.dropout2(x)
        
        x = F.gelu(self.dense3(x))
        x = self.bn3(x)
        
        x = self.output_layer(x)
        return x


class CNNModel(nn.Module):
    """
    1D Convolutional Neural Network for tabular data.
    
    Treats the features as a 1D sequence and applies convolutional filters.
    """
    def __init__(self, n_features, dropout_rate=0.3):
        super(CNNModel, self).__init__()
        # Reshape for CNN
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        
        # Flatten and dense layers
        self.dense1 = nn.Linear(64 * n_features, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.output_layer = nn.Linear(128, 1)
    
    def forward(self, x):
        # Reshape input for CNN
        x = x.unsqueeze(1)  # Add channel dimension
        
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.dense1(x))
        x = self.bn3(x)
        x = self.dropout(x)
        
        x = self.output_layer(x)
        return x


class FFNWithPosEncoding(nn.Module):
    """
    Feed-Forward Network with Positional Encoding for tabular data.
    
    Adds positional encoding to each feature to provide 'location' information.
    """
    def __init__(self, n_features, dropout_rate=0.1):
        super(FFNWithPosEncoding, self).__init__()
        self.embed_dim = 32  # Embedding size for each token
        self.ff_dim = 32  # Hidden layer size in feed forward network
        
        # Feed forward network
        self.ffn = nn.Sequential(
            nn.Linear(1, self.ff_dim),
            nn.ReLU(),
            nn.Linear(self.ff_dim, self.embed_dim)
        )
        
        # Layer normalization
        self.layernorm = nn.LayerNorm(self.embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Positional embedding
        self.pos_emb = nn.Embedding(n_features, self.embed_dim)
        
        # Register buffer for positions
        positions = torch.arange(0, n_features).long()
        self.register_buffer('positions', positions)
        
        # Output layer
        self.output_layer = nn.Linear(n_features * self.embed_dim, 1)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Expand dimensions of inputs to match embedding output
        x = x.unsqueeze(2)  # Shape becomes [batch_size, n_features, 1]
        
        # Pass inputs through the feed forward network
        x = self.ffn(x)  # Shape becomes [batch_size, n_features, embed_dim]
        
        # Get positional embeddings
        pos_encoding = self.pos_emb(self.positions)  # Shape [n_features, embed_dim]
        
        # Expand pos_encoding to match the batch size of inputs
        pos_encoding = pos_encoding.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Add dropout
        x = self.dropout(x)
        
        # Add positional encodings
        x = x + pos_encoding
        
        # Apply layer normalization
        x = self.layernorm(x)
        
        # Flatten the output for the classifier
        x = x.reshape(batch_size, -1)
        
        # Output
        x = self.output_layer(x)
        return x


class EntropyLayer(nn.Module):
    """
    Physics-informed Entropy Layer based on Boltzmann-Gibbs entropy formula.
    
    Models landscape complexity and wildfire susceptibility.
    """
    def __init__(self, n_landcover, m_env_factors):
        super(EntropyLayer, self).__init__()
        # Trainable scaling constant for entropy term
        self.k = nn.Parameter(torch.ones(1))
        # Trainable weights for environmental factors
        self.alpha = nn.Parameter(torch.ones(m_env_factors))
        self.n_landcover = n_landcover
        self.m_env_factors = m_env_factors

    def forward(self, inputs):
        # Get the actual dimensions of the input
        input_shape = inputs.size()
        n_features = input_shape[1]
        
        # Adjust n_landcover if it's larger than the input size
        n_landcover_adjusted = min(self.n_landcover, n_features)
        
        # Split input into land cover proportions and environmental factors
        p_i = F.softmax(inputs[:, :n_landcover_adjusted], dim=-1)
        f_j = inputs[:, n_landcover_adjusted:]
        
        # Get the actual size of f_j
        f_j_size = f_j.size(1)
        
        # Use only as many alpha values as there are features in f_j
        alpha_adjusted = self.alpha[:f_j_size]
        
        # Calculate entropy term (landscape diversity)
        entropy_term = -self.k * torch.sum(
                    p_i * torch.log(p_i + 1e-10), dim=-1)
        
        # Calculate environmental influence term
        env_term = torch.sum(alpha_adjusted * f_j, dim=-1)
        
        # Return combined entropy score (scalar per sample)
        return (entropy_term + env_term).unsqueeze(1)


class PhysicsEmbeddedEntropyModel(nn.Module):
    """
    Physics-Embedded Entropy Model for wildfire prediction.
    
    Integrates principles from statistical mechanics with deep learning through
    an entropy layer that models landscape complexity.
    """
    def __init__(self, n_features, n_landcover=4, m_env_factors=None):
        super(PhysicsEmbeddedEntropyModel, self).__init__()
        if m_env_factors is None:
            m_env_factors = min(300, n_features - n_landcover)
        
        # FFN Branch
        self.ffn_branch1 = nn.Linear(n_features, 256)
        self.ffn_bn1 = nn.BatchNorm1d(256)
        self.ffn_dropout = nn.Dropout(0.3)
        self.ffn_branch2 = nn.Linear(256, 128)
        self.ffn_bn2 = nn.BatchNorm1d(128)
        
        # 1D CNN Branch
        self.cnn_branch = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.cnn_bn = nn.BatchNorm1d(32)
        self.cnn_dense = nn.Linear(32 * n_features, 128)
        self.cnn_bn2 = nn.BatchNorm1d(128)
        
        # PMFFNN Branch
        self.split_size = n_features // 3
        # Branch 1
        self.pmffnn_branch1 = nn.Linear(self.split_size, 64)
        self.pmffnn_bn1 = nn.BatchNorm1d(64)
        
        # Branch 2
        self.pmffnn_branch2 = nn.Linear(self.split_size, 64)
        self.pmffnn_bn2 = nn.BatchNorm1d(64)
        
        # Branch 3 (might be slightly larger if n_features isn't divisible by 3)
        self.pmffnn_branch3 = nn.Linear(n_features - 2*self.split_size, 64)
        self.pmffnn_bn3 = nn.BatchNorm1d(64)
        
        # PMFFNN integration
        self.pmffnn_dense = nn.Linear(64 * 3, 128)
        self.pmffnn_bn4 = nn.BatchNorm1d(128)
        
        # Integration Network
        self.integrated_dense1 = nn.Linear(128 * 3, 512)
        self.integrated_bn1 = nn.BatchNorm1d(512)
        self.integrated_dropout = nn.Dropout(0.3)
        self.integrated_dense2 = nn.Linear(512, 256)
        self.integrated_bn2 = nn.BatchNorm1d(256)
        
        # Physics-Embedded Entropy Layer
        self.entropy_layer = EntropyLayer(n_landcover, m_env_factors)
        
        # Multi-path classification with sigmoid layers
        self.sigmoid_branch1 = nn.Linear(128 + 1, 128)
        self.sigmoid_branch2 = nn.Linear(128 + 1, 128)
        self.sigmoid_branch3 = nn.Linear(128 + 1, 128)
        
        # Output layer
        self.output_layer = nn.Linear(128 * 3, 1)
        
    def forward(self, x):
        # FFN Branch
        ffn = F.gelu(self.ffn_branch1(x))
        ffn = self.ffn_bn1(ffn)
        ffn = self.ffn_dropout(ffn)
        ffn = F.gelu(self.ffn_branch2(ffn))
        ffn_out = self.ffn_bn2(ffn)
        
        # 1D CNN Branch
        cnn_input = x.unsqueeze(1)  # Reshape for CNN input
        cnn = F.selu(self.cnn_branch(cnn_input))
        cnn = self.cnn_bn(cnn)
        cnn = cnn.view(cnn.size(0), -1)  # Flatten
        cnn = F.selu(self.cnn_dense(cnn))
        cnn_out = self.cnn_bn2(cnn)
        
        # PMFFNN Branch - split features into 3 groups
        branch1_input = x[:, :self.split_size]
        branch2_input = x[:, self.split_size:2*self.split_size]
        branch3_input = x[:, 2*self.split_size:]
        
        branch1 = F.selu(self.pmffnn_branch1(branch1_input))
        branch1 = self.pmffnn_bn1(branch1)
        
        branch2 = F.selu(self.pmffnn_branch2(branch2_input))
        branch2 = self.pmffnn_bn2(branch2)
        
        branch3 = F.selu(self.pmffnn_branch3(branch3_input))
        branch3 = self.pmffnn_bn3(branch3)
        
        # Concatenate PMFFNN branches
        pmffnn_concat = torch.cat([branch1, branch2, branch3], dim=1)
        pmffnn = F.selu(self.pmffnn_dense(pmffnn_concat))
        pmffnn_out = self.pmffnn_bn4(pmffnn)
        
        # Concatenate all branch outputs
        concat = torch.cat([ffn_out, cnn_out, pmffnn_out], dim=1)
        
        # Integration Network
        integrated = F.gelu(self.integrated_dense1(concat))
        integrated = self.integrated_bn1(integrated)
        integrated = self.integrated_dropout(integrated)
        integrated = F.gelu(self.integrated_dense2(integrated))
        integrated = self.integrated_bn2(integrated)
        
        # Physics-Embedded Entropy Layer
        entropy_out = self.entropy_layer(integrated)
        
        # Residual connection from FFN branch
        combined = torch.cat([entropy_out, ffn_out], dim=1)
        
        # Multi-path classification with sigmoid layers
        sig_branch1 = torch.sigmoid(self.sigmoid_branch1(combined))
        sig_branch2 = torch.sigmoid(self.sigmoid_branch2(combined))
        sig_branch3 = torch.sigmoid(self.sigmoid_branch3(combined))
        
        # Concatenate sigmoid branches
        sig_concat = torch.cat([sig_branch1, sig_branch2, sig_branch3], dim=1)
        
        # Output
        output = self.output_layer(sig_concat)
        
        return output


class CustomGemmaLayer(nn.Module):
    """
    A custom transformer layer that mimics Gemma's functionality
    without requiring position embeddings.
    """
    def __init__(self, hidden_size=1152, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Self attention
        self.layernorm_before = nn.LayerNorm(hidden_size)
        self.self_attn = nn.MultiheadAttention(
            hidden_size, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        # Feed forward
        self.layernorm_after = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        
    def forward(self, hidden_states, **kwargs):
        # First residual block: self-attention
        residual = hidden_states
        hidden_states = self.layernorm_before(hidden_states)
        attn_output, _ = self.self_attn(hidden_states, hidden_states, hidden_states)
        hidden_states = residual + attn_output
        
        # Second residual block: feed-forward
        residual = hidden_states
        hidden_states = self.layernorm_after(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states
