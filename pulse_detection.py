import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridCNNLSTM(nn.Module):
    """Hybrid CNN-LSTM model witth attention mechanism"""
    def __init__(self,cnn_channels = 32, lstm_hidden = 64, attention_heads = 4):
        super(HybridCNNLSTM, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, cnn_channels, kernel_size=5, padding=2),
            nn.LeakyReLU(),
            nn.MaxPool1d(2)
        )
        self.lstm = nn.LSTM(cnn_channels,lstm_hidden,batch_first=True)
        self.attention = nn.MultiheadAttention(embed_dim=lstm_hidden, num_heads=attention_heads)
        self.fc = nn.Linear(lstm_hidden, 3)
        self.dropout = nn.Dropout(0,2) #Adding dropout for Regularization

    def forward(self, x):
        """Forward pass with CNN, LSTM and attention Layers"""
        x = x.unsqueeze(1) #Adding channel dimension
        x = self.conv(x) #Convolutional feature extraction
        x = x.permute(0, 2, 1) #Permuting for LSTM [Batch, Seq, Channels]
        lstm_out, _ = self.lstm(x) #LSTM temporal processing
        lstm_out = self.dropout(lstm_out) #Adding dropout

        #Multi-head self-attention mechanisum
        attn_output, attn_weights =self.attention(lstm_out, lstm_out, lstm_out)
        attn_output = self.dropout(attn_output) #pool the attention output to form a context vector
        
        context_vector = torch.sum(attn_output, dim = 1) # Global average pooling over time steps

        return self.fc(context_vector)