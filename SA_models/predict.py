import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import h5py
import scipy.stats as stats

# Function to load dataset from file
'''
def load_data(file_path):
    sequences = []
    sequences_onehot = []
    values = []
    with open(file_path, 'r') as file:
        for line in file:
            sequence, value = line.strip().split()  # Assuming columns are space-separated
            encoded_sequence = []
            for ch in sequence:
                if ch == 'N':
                    encoded_sequence.append([0.5, 0.5, 0.5, 0.5])
                else:
                    one_hot = [0] * 4
                    one_hot["ACGT".index(ch)] = 1
                    encoded_sequence.append(one_hot)
            sequences.append(sequence)
            sequences_onehot.append(encoded_sequence)
            values.append(float(value))
    return torch.tensor(sequences_onehot, dtype=torch.float32), torch.tensor(values, dtype=torch.float32), sequences
'''



def load_h5_data(file_path):
    with h5py.File(file_path, "r") as h5_file:
        # Access datasets inside 'data' group
        c0_y = h5_file["data/c0_y"][:].squeeze()       # Load as NumPy array
        s_x = h5_file["data/s_x"][:]
        sequence = h5_file["data/sequence"][:]
        decoded_sequences = [seq.decode('utf-8') for seq in sequence]
        # Access datasets inside 'targets' group
        id_data = h5_file["targets/id"][:]
        name_data = h5_file["targets/name"][:]
        print("c0_y shape:", c0_y.shape)
        print("s_x shape:", s_x.shape)
        print("id shape:", id_data.shape)
        print("name shape:", name_data.shape)
    return torch.tensor(s_x, dtype=torch.float32), torch.tensor(c0_y, dtype=torch.float32), decoded_sequences


# Positional encoding class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)
    def forward(self, x):
        return x + self.encoding[:, :x.size(1), :].to(x.device)

# Transformer model
class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, nlayer, max_len):
        super(SimpleTransformer, self).__init__()
        self.linear_in = nn.Linear(input_dim, d_model)  # (batch_size, seq_len, vocab_size) -> (batch_size, seq_len, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
            num_layers=nlayer
        )
        self.fc = nn.Linear(d_model, 1)  # (batch_size, d_model) -> (batch_size, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.linear_in(x)  # Linear transformation from one-hot to dense (batch_size, seq_len, d_model)
        x = self.positional_encoding(x)  # Add positional encoding (batch_size, seq_len, d_model)
        x = self.transformer(x)  # Transformer block (batch_size, seq_len, d_model)
        x = x.mean(dim=1)  # Global average pooling (batch_size, d_model)
        x = self.fc(x)
        return self.sigmoid(x)

# Visualization function for attention map
def visualize_attention(model, sequence_onehot, sequence, device):
    model.eval()
    linear_in = model.linear_in
    positional_encoding = model.positional_encoding
    self_atten_1 = model.transformer.layers[0].self_attn
    sequence_onehot = sequence_onehot.to(device)
    with torch.no_grad():
        lineared = linear_in(sequence_onehot)
        positioned = positional_encoding(lineared)  # Add positional encoding (1, seq_len, d_model)
        attn_output, attn_weights = self_atten_1(
            query=positioned, key=positioned, value=positioned, need_weights=True
        )
        # pdb.set_trace()
    attn_weights = attn_weights.squeeze(0).cpu().numpy()  # (seq_len, seq_len)
    plt.imshow(attn_weights, cmap="viridis")
    plt.colorbar()
    plt.title("Attention Map")
    x = [i for i in range(len(sequence))]
    ticks = [nucleotide for nucleotide in sequence]
    plt.xticks(x,ticks)
    plt.yticks(x,ticks)
    plt.xlabel("Sequence Positions")
    plt.ylabel("Sequence Positions")
    figname = 'predict_h5_attention_'+sequence+'.png'
    print(figname)
    plt.savefig(figname)
    plt.close()


# Pearson Correlation Function
def pearson_correlation(preds, targets):
    preds = preds.cpu().numpy().flatten()
    targets = targets.cpu().numpy().flatten()
    return pearsonr(preds, targets)[0]  # Return Pearson coefficient


# Load test data
test_sequences_onehot, test_values, test_sequences = load_h5_data("data/SELEX_canonical/Ubx/Ubx_test.h5")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model parameters (must match the trained model)
input_dim = 4  
d_model = 128  
nlayer = 2
nhead = 8
max_len = 14

# Load trained model
model = SimpleTransformer(input_dim, d_model, nhead, nlayer, max_len).to(device)
model.load_state_dict(torch.load("Ubx_canonical_models_gridsearch_0219/lr0.001_batch256_nlayer2_nhead8_dmodel128.pth"))
model.eval()
print("Model Loaded: Ubx_canonical_models_gridsearch_0219/lr0.001_batch256_nlayer2_nhead8_dmodel128.pth")


# Move test data to device
test_sequences_onehot, test_values = test_sequences_onehot.to(device), test_values.to(device)

# Run predictions
with torch.no_grad():
    predictions = model(test_sequences_onehot).squeeze()

# calculate R2
_, _, r_value, _, _ = stats.linregress(predictions, test_values)
# Compute Pearson correlation
pearson_corr = pearson_correlation(predictions, test_values)
print(f"Pearson Correlation on Test Set: {pearson_corr:.4f}, R2 on test set: {r_value**2}")

# Attention visualization (for the first test sequence)
sample_sequence_onehot = test_sequences_onehot[2].unsqueeze(0)  # Add batch dimension
sample_sequence = test_sequences[2]
# visualize_attention(model, sample_sequence_onehot, sample_sequence, device)
