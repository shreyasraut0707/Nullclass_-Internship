"""
Sequence-to-Sequence model with attention mechanism for translation.
This implements an encoder-decoder architecture with Bahdanau attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class Encoder(nn.Module):
    """Encoder with bidirectional LSTM."""
    
    def __init__(self, input_size, embedding_dim, hidden_dim, num_layers, dropout):
        super(Encoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src):
        """
        Args:
            src: [batch_size, src_len]
        Returns:
            outputs: [batch_size, src_len, hidden_dim * 2]
            hidden: tuple of [num_layers * 2, batch_size, hidden_dim]
        """
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell


class Attention(nn.Module):
    """Bahdanau attention mechanism."""
    
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super(Attention, self).__init__()
        
        # Encoder is bidirectional so we multiply hidden_dim by 2
        self.attention = nn.Linear(encoder_hidden_dim * 2 + decoder_hidden_dim, decoder_hidden_dim)
        self.v = nn.Linear(decoder_hidden_dim, 1, bias=False)
    
    def forward(self, hidden, encoder_outputs):
        """
        Args:
            hidden: [batch_size, decoder_hidden_dim]
            encoder_outputs: [batch_size, src_len, encoder_hidden_dim * 2]
        Returns:
            attention_weights: [batch_size, src_len]
        """
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # Repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        # Calculate attention energy
        energy = torch.tanh(self.attention(torch.cat((hidden, encoder_outputs), dim=2)))
        
        attention = self.v(energy).squeeze(2)
        
        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    """Decoder with attention mechanism."""
    
    def __init__(self, output_size, embedding_dim, encoder_hidden_dim, 
                 decoder_hidden_dim, num_layers, dropout):
        super(Decoder, self).__init__()
        
        self.output_size = output_size
        self.decoder_hidden_dim = decoder_hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(output_size, embedding_dim, padding_idx=0)
        self.attention = Attention(encoder_hidden_dim, decoder_hidden_dim)
        
        # LSTM input is embedding + context vector (encoder_hidden_dim * 2)
        self.lstm = nn.LSTM(
            embedding_dim + encoder_hidden_dim * 2,
            decoder_hidden_dim,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Linear(decoder_hidden_dim, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input, hidden, cell, encoder_outputs):
        """
        Args:
            input: [batch_size]
            hidden: [num_layers, batch_size, decoder_hidden_dim]
            cell: [num_layers, batch_size, decoder_hidden_dim]
            encoder_outputs: [batch_size, src_len, encoder_hidden_dim * 2]
        Returns:
            prediction: [batch_size, output_size]
            hidden: [num_layers, batch_size, decoder_hidden_dim]
            cell: [num_layers, batch_size, decoder_hidden_dim]
        """
        input = input.unsqueeze(1)  # [batch_size, 1]
        embedded = self.dropout(self.embedding(input))  # [batch_size, 1, embedding_dim]
        
        # Calculate attention weights using top layer hidden state
        attention_weights = self.attention(hidden[-1], encoder_outputs)  # [batch_size, src_len]
        attention_weights = attention_weights.unsqueeze(1)  # [batch_size, 1, src_len]
        
        # Calculate context vector
        context = torch.bmm(attention_weights, encoder_outputs)  # [batch_size, 1, encoder_hidden_dim * 2]
        
        # Concatenate embedding and context
        lstm_input = torch.cat((embedded, context), dim=2)
        
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        
        # Make prediction
        prediction = self.fc(output.squeeze(1))
        
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    """Complete Sequence-to-Sequence model."""
    
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        """
        Args:
            src: [batch_size, src_len]
            tgt: [batch_size, tgt_len]
            teacher_forcing_ratio: probability of using teacher forcing
        Returns:
            outputs: [batch_size, tgt_len, output_size]
        """
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.output_size
        
        # Store outputs
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)
        
        # Encode source
        encoder_outputs, hidden, cell = self.encoder(src)
        
        # Prepare decoder hidden state
        # Encoder is bidirectional, so we need to combine forward and backward states
        hidden = self._combine_bidirectional(hidden, self.decoder.num_layers)
        cell = self._combine_bidirectional(cell, self.decoder.num_layers)
        
        # First input to decoder is SOS token
        input = tgt[:, 0]
        
        for t in range(1, tgt_len):
            # Decode one step
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            
            outputs[:, t, :] = output
            
            # Teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            
            input = tgt[:, t] if teacher_force else top1
        
        return outputs
    
    def _combine_bidirectional(self, state, num_layers):
        """Combine bidirectional encoder states for decoder."""
        # state is [num_layers * 2, batch_size, hidden_dim]
        # We need [num_layers, batch_size, decoder_hidden_dim]
        batch_size = state.shape[1]
        hidden_dim = state.shape[2]
        
        # Reshape and combine
        state = state.view(num_layers, 2, batch_size, hidden_dim)
        state = torch.cat([state[:, 0, :, :], state[:, 1, :, :]], dim=2)
        
        # Project back to decoder hidden dim if needed
        if state.shape[2] != self.decoder.decoder_hidden_dim:
            state = state[:, :, :self.decoder.decoder_hidden_dim]
        
        # Ensure contiguous memory layout
        return state.contiguous()


def init_weights(model):
    """Initialize model weights."""
    for name, param in model.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model creation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    INPUT_DIM = 10000
    OUTPUT_DIM = 10000
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    ENC_HID_DIM = 512
    DEC_HID_DIM = 512
    NUM_LAYERS = 2
    DROPOUT = 0.3
    
    encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, NUM_LAYERS, DROPOUT)
    decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, NUM_LAYERS, DROPOUT)
    
    model = Seq2Seq(encoder, decoder, device).to(device)
    init_weights(model)
    
    print(f"Model created with {count_parameters(model):,} trainable parameters")
    
    # Test forward pass
    src = torch.randint(0, INPUT_DIM, (4, 10)).to(device)
    tgt = torch.randint(0, OUTPUT_DIM, (4, 12)).to(device)
    
    output = model(src, tgt)
    print(f"Output shape: {output.shape}")
