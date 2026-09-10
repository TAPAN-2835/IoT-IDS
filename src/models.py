import torch
import torch.nn as nn

class CNN1D(nn.Module):
    """
    1D CNN Baseline
    Input -> Conv1D -> BatchNorm -> ReLU -> MaxPooling -> Dropout -> Dense -> Output
    Note: CNN is tested empirically on ordered flow features. 
    We do not claim feature ordering is naturally spatial.
    """
    def __init__(self, input_dim, num_classes=1):
        super(CNN1D, self).__init__()
        
        # input_dim corresponds to the number of features.
        # We treat the tabular vector as 1 channel, sequence length = input_dim.
        # Shape expected by Conv1d: (batch_size, in_channels, sequence_length) -> (N, 1, input_dim)
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.2)
        )
        
        # Calculate size after Conv and MaxPool
        # Length becomes input_dim // 2
        linear_input_size = 32 * (input_dim // 2)
        
        self.dense = nn.Sequential(
            nn.Linear(linear_input_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # x shape: (N, input_dim)
        x = x.unsqueeze(1) # (N, 1, input_dim)
        x = self.conv(x) # (N, 32, input_dim // 2)
        x = x.view(x.size(0), -1) # Flatten (N, 32 * (input_dim // 2))
        x = self.dense(x)
        return x

class GRUBaseline(nn.Module):
    """
    GRU Baseline
    Limitation Documented: The preprocessed dataset does not provide defensible 
    temporal sequence information since IP and timestamps were removed as leakage.
    We DO NOT create fake temporal sequences by randomly reshaping independent rows.
    This GRU experiment is explicitly scoped as exploratory, treating the feature vector
    as a single time step sequence (seq_len=1, input_size=input_dim).
    """
    def __init__(self, input_dim, num_classes=1):
        super(GRUBaseline, self).__init__()
        
        # Sequence length = 1, input_size = input_dim
        self.gru = nn.GRU(input_size=input_dim, hidden_size=64, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.dense = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # x shape: (N, input_dim)
        x = x.unsqueeze(1) # (N, seq_len=1, input_size=input_dim)
        out, _ = self.gru(x) # out shape: (N, 1, 64)
        out = out[:, -1, :] # (N, 64)
        out = self.dropout(out)
        out = self.dense(out)
        return out

class CNN_GRU(nn.Module):
    """
    Proposed Hybrid CNN-GRU
    Input -> Conv1D -> BatchNorm -> ReLU -> MaxPooling -> Dropout -> GRU -> Dense -> Dropout -> Output
    """
    def __init__(self, input_dim, num_classes=1):
        super(CNN_GRU, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.2)
        )
        
        # Output of conv is (N, 32, input_dim // 2)
        # We transpose this to (N, input_dim // 2, 32) so seq_len = input_dim // 2, input_size = 32
        
        self.gru = nn.GRU(input_size=32, hidden_size=64, batch_first=True)
        self.dense = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # x shape: (N, input_dim)
        x = x.unsqueeze(1) # (N, 1, input_dim)
        x = self.conv(x) # (N, 32, input_dim // 2)
        
        # Transpose for GRU: (batch, seq_len, features)
        x = x.transpose(1, 2) # (N, input_dim // 2, 32)
        
        out, _ = self.gru(x) # (N, input_dim // 2, 64)
        out = out[:, -1, :] # Take last hidden state (N, 64)
        out = self.dense(out)
        return out
