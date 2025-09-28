import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math

class Encoder(nn.Module):
    """
    Enhanced BiLSTM Encoder with attention support.
    """
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(
            emb_dim,
            enc_hid_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Transform concatenated bidirectional hidden state to decoder hidden size
        self.fc_hidden = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.fc_cell = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training stability."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, src, src_len):
        # src: [batch_size, seq_len]
        embedded = self.dropout(self.embedding(src))  # [batch_size, seq_len, emb_dim]
        
        # Pack sequences for efficiency
        packed_embedded = pack_padded_sequence(
            embedded, src_len.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # BiLSTM forward pass
        packed_outputs, (hidden, cell) = self.rnn(packed_embedded)
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        
        # hidden, cell: [num_layers * 2, batch_size, enc_hid_dim] (due to bidirectional)
        # Reshape: [num_layers, 2, batch_size, enc_hid_dim]
        batch_size = hidden.size(1)
        hidden = hidden.view(self.num_layers, 2, batch_size, self.enc_hid_dim)
        cell = cell.view(self.num_layers, 2, batch_size, self.enc_hid_dim)
        
        # Take the last layer and concatenate forward and backward
        final_hidden = torch.cat([hidden[-1, 0], hidden[-1, 1]], dim=1)  # [batch_size, enc_hid_dim * 2]
        final_cell = torch.cat([cell[-1, 0], cell[-1, 1]], dim=1)        # [batch_size, enc_hid_dim * 2]
        
        # Transform to decoder dimensions
        final_hidden = torch.tanh(self.fc_hidden(final_hidden))  # [batch_size, dec_hid_dim]
        final_cell = torch.tanh(self.fc_cell(final_cell))        # [batch_size, dec_hid_dim]
        
        return outputs, (final_hidden, final_cell)

class Attention(nn.Module):
    """
    Bahdanau Attention mechanism for better translation quality.
    """
    def __init__(self, enc_hid_dim, dec_hid_dim, attn_dim):
        super().__init__()
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.attn_dim = attn_dim
        
        self.attn = nn.Linear(enc_hid_dim * 2 + dec_hid_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)
        
    def forward(self, encoder_outputs, decoder_hidden, src_mask=None):
        """
        encoder_outputs: [batch_size, src_len, enc_hid_dim * 2]
        decoder_hidden: [batch_size, dec_hid_dim]
        src_mask: [batch_size, src_len] (optional)
        """
        batch_size = encoder_outputs.size(0)
        src_len = encoder_outputs.size(1)
        
        # Repeat decoder hidden state for each source position
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        # Calculate attention scores
        energy = torch.tanh(self.attn(torch.cat((encoder_outputs, decoder_hidden), dim=2)))
        attention_scores = self.v(energy).squeeze(2)  # [batch_size, src_len]
        
        # Apply source mask if provided
        if src_mask is not None:
            attention_scores = attention_scores.masked_fill(src_mask == 0, -1e10)
        
        # Calculate attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, src_len]
        
        # Calculate context vector
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)  # [batch_size, 1, enc_hid_dim * 2]
        context = context.squeeze(1)  # [batch_size, enc_hid_dim * 2]
        
        return context, attention_weights

class Decoder(nn.Module):
    """
    Enhanced LSTM Decoder with attention mechanism.
    """
    def __init__(self, output_dim, emb_dim, dec_hid_dim, enc_hid_dim, num_layers=4, dropout=0.3, attention=True):
        super().__init__()
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dec_hid_dim = dec_hid_dim
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(
            emb_dim + (enc_hid_dim * 2 if attention else 0),
            dec_hid_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        if attention:
            self.attention_layer = Attention(enc_hid_dim, dec_hid_dim, dec_hid_dim)
            self.fc_out = nn.Linear(dec_hid_dim + enc_hid_dim * 2, output_dim)
        else:
            self.fc_out = nn.Linear(dec_hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training stability."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, input_token, hidden, cell, encoder_outputs=None, src_mask=None):
        # input_token: [batch_size]
        # hidden: [num_layers, batch_size, dec_hid_dim]
        # cell: [num_layers, batch_size, dec_hid_dim]
        
        input_token = input_token.unsqueeze(1)  # [batch_size, 1]
        embedded = self.dropout(self.embedding(input_token))  # [batch_size, 1, emb_dim]
        
        if self.attention and encoder_outputs is not None:
            # Get attention context
            context, attention_weights = self.attention_layer(
                encoder_outputs, hidden[-1], src_mask
            )
            context = context.unsqueeze(1)  # [batch_size, 1, enc_hid_dim * 2]
            
            # Concatenate embedded input with context
            rnn_input = torch.cat((embedded, context), dim=2)
        else:
            rnn_input = embedded
            attention_weights = None
        
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        # output: [batch_size, 1, dec_hid_dim]
        
        if self.attention and encoder_outputs is not None:
            # Concatenate output with context for final prediction
            context = context.squeeze(1)  # [batch_size, enc_hid_dim * 2]
            prediction = self.fc_out(torch.cat((output.squeeze(1), context), dim=1))
        else:
            prediction = self.fc_out(output.squeeze(1))
        
        return prediction, hidden, cell, attention_weights

class Seq2Seq(nn.Module):
    """
    Enhanced Seq2Seq model with attention and improved features.
    """
    def __init__(self, encoder, decoder, device, attention=True):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.attention = attention
    
    def create_initial_decoder_state(self, encoder_hidden, encoder_cell, batch_size):
        """
        Create initial hidden and cell states for decoder.
        """
        dec_num_layers = self.decoder.num_layers
        dec_hid_dim = self.decoder.dec_hid_dim
        
        # Initialize all decoder layers with the encoder's final state
        hidden = encoder_hidden.unsqueeze(0).expand(dec_num_layers, batch_size, dec_hid_dim).contiguous()
        cell = encoder_cell.unsqueeze(0).expand(dec_num_layers, batch_size, dec_hid_dim).contiguous()
        
        return hidden, cell
    
    def create_src_mask(self, src, src_len):
        """Create source mask for attention."""
        batch_size = src.size(0)
        src_len_max = src.size(1)
        mask = torch.zeros(batch_size, src_len_max, dtype=torch.bool, device=self.device)
        
        for i, length in enumerate(src_len):
            mask[i, :length] = True
        
        return mask
    
    def forward(self, src, src_len, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        # Store outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # Encode the source sequence
        encoder_outputs, (enc_hidden, enc_cell) = self.encoder(src, src_len)
        
        # Create source mask for attention
        src_mask = self.create_src_mask(src, src_len) if self.attention else None
        
        # Initialize decoder hidden states
        hidden, cell = self.create_initial_decoder_state(enc_hidden, enc_cell, batch_size)
        
        # First input to decoder is <sos> token
        input_token = trg[:, 0]  # [batch_size]
        
        # Decode step by step
        for t in range(1, trg_len):
            # Forward pass through decoder
            if self.attention:
                output, hidden, cell, attention_weights = self.decoder(
                    input_token, hidden, cell, encoder_outputs, src_mask
                )
            else:
                output, hidden, cell, _ = self.decoder(input_token, hidden, cell)
            
            # Store output
            outputs[:, t] = output
            
            # Get the highest predicted token
            top1 = output.argmax(1)
            
            # Use teacher forcing or not
            input_token = trg[:, t] if torch.rand(1).item() < teacher_forcing_ratio else top1
        
        return outputs
    
    def translate_sentence(self, src_vocab, tgt_idx2char, sentence, max_len=100):
        """
        Translate a single sentence with beam search (simplified greedy).
        """
        self.eval()
        with torch.no_grad():
            # Tokenize input
            tokenized = []
            for char in sentence:
                if char in src_vocab:
                    tokenized.append(src_vocab[char])
                else:
                    tokenized.append(src_vocab.get('<unk>', 3))
            
            if not tokenized:
                return "Invalid input"
            
            # Create tensors
            src_tensor = torch.LongTensor(tokenized).unsqueeze(0).to(self.device)
            src_len = torch.LongTensor([len(tokenized)]).to(self.device)
            
            # Encode
            encoder_outputs, (enc_hidden, enc_cell) = self.encoder(src_tensor, src_len)
            
            # Create source mask
            src_mask = self.create_src_mask(src_tensor, src_len) if self.attention else None
            
            # Initialize decoder states
            batch_size = src_tensor.size(0)
            hidden, cell = self.create_initial_decoder_state(enc_hidden, enc_cell, batch_size)
            
            # Generate translation
            trg_indexes = [1]  # <sos> token
            max_len = min(len(sentence) * 2, max_len)
            
            for _ in range(max_len):
                trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(self.device)
                
                if self.attention:
                    output, hidden, cell, _ = self.decoder(
                        trg_tensor, hidden, cell, encoder_outputs, src_mask
                    )
                else:
                    output, hidden, cell, _ = self.decoder(trg_tensor, hidden, cell)
                
                pred_token = output.argmax(1).item()
                trg_indexes.append(pred_token)
                
                if pred_token == 2:  # <eos>
                    break
            
            # Convert to string
            translation = []
            for idx in trg_indexes[1:-1]:  # Skip <sos> and <eos>
                if idx in tgt_idx2char and idx not in [0, 1, 2, 3]:  # Skip special tokens
                    translation.append(tgt_idx2char[idx])
            
            return ''.join(translation)
    
    def count_parameters(self):
        """Count total number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self):
        """Get model architecture information."""
        return {
            'encoder_layers': self.encoder.num_layers,
            'decoder_layers': self.decoder.num_layers,
            'encoder_hidden_dim': self.encoder.enc_hid_dim,
            'decoder_hidden_dim': self.decoder.dec_hid_dim,
            'attention': self.attention,
            'total_parameters': self.count_parameters()
        }

# ================================
# MODEL FACTORY FUNCTIONS
# ================================

def create_model(input_dim, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, 
                enc_layers=2, dec_layers=4, dropout=0.3, attention=True, device='cpu'):
    """
    Factory function to create a complete Seq2Seq model.
    """
    encoder = Encoder(input_dim, emb_dim, enc_hid_dim, dec_hid_dim, enc_layers, dropout)
    decoder = Decoder(output_dim, emb_dim, dec_hid_dim, enc_hid_dim, dec_layers, dropout, attention)
    model = Seq2Seq(encoder, decoder, device, attention)
    
    return model

def create_experiment_models(device='cpu'):
    """
    Create different model configurations for hyperparameter experiments.
    """
    # Base configuration
    base_config = {
        'input_dim': 60,
        'output_dim': 40,
        'emb_dim': 256,
        'enc_hid_dim': 512,
        'dec_hid_dim': 512,
        'enc_layers': 2,
        'dec_layers': 4,
        'dropout': 0.3,
        'attention': True,
        'device': device
    }
    
    # Experiment configurations
    experiments = {
        'baseline': base_config,
        
        'small_emb': {**base_config, 'emb_dim': 128},
        'large_emb': {**base_config, 'emb_dim': 512},
        
        'small_hidden': {**base_config, 'enc_hid_dim': 256, 'dec_hid_dim': 256},
        'large_hidden': {**base_config, 'enc_hid_dim': 1024, 'dec_hid_dim': 1024},
        
        'single_enc': {**base_config, 'enc_layers': 1},
        'triple_enc': {**base_config, 'enc_layers': 3},
        'quad_enc': {**base_config, 'enc_layers': 4},
        
        'double_dec': {**base_config, 'dec_layers': 2},
        'triple_dec': {**base_config, 'dec_layers': 3},
        'penta_dec': {**base_config, 'dec_layers': 5},
        
        'low_dropout': {**base_config, 'dropout': 0.1},
        'high_dropout': {**base_config, 'dropout': 0.5},
        
        'no_attention': {**base_config, 'attention': False}
    }
    
    return experiments

if __name__ == "__main__":
    # Test the enhanced model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create a test model
    model = create_model(
        input_dim=60, output_dim=40, emb_dim=256, 
        enc_hid_dim=512, dec_hid_dim=512,
        enc_layers=2, dec_layers=4, dropout=0.3,
        attention=True, device=device
    )
    
    print("🚀 Enhanced Model Created Successfully!")
    print(f"📊 Model Info: {model.get_model_info()}")
    
    # Test forward pass
    batch_size, src_len, trg_len = 2, 10, 8
    src = torch.randint(0, 60, (batch_size, src_len)).to(device)
    src_len_tensor = torch.tensor([src_len, src_len-2]).to(device)
    trg = torch.randint(0, 40, (batch_size, trg_len)).to(device)
    
    output = model(src, src_len_tensor, trg)
    print(f"✅ Forward pass successful! Output shape: {output.shape}")
