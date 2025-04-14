import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import h5py
import sys
import os
from matplotlib.transforms import Affine2D

if len(sys.argv) > 7:
    TFname = sys.argv[1]
    ModelType = sys.argv[2]
    SeqLen = int(sys.argv[3])
    DateMark = sys.argv[4]
    Nlayer = int(sys.argv[5])
    Nhead = int(sys.argv[6])
    pred_data_name = sys.argv[7] # used to load prediction/data/pred_data_file.txt, and save pred values into prediction/pred_data_name.txt
else:
    print("Please provide a TFname, ModelType, Seqlen ... etc as command-line arguments")
    sys.exit(1)

trained_model = f'{TFname}_{ModelType}_models_gridsearch_{DateMark}/lr0.001_batch256_nlayer{Nlayer}_nhead{Nhead}_dmodel128.pth'
pred_data_file = f'prediction/data/{pred_data_name}.txt'

# create folders for prediction result:
save_result_dir = f'prediction/{TFname}_{ModelType}_{DateMark}_nlayer{Nlayer}_nhead{Nhead}/'
os.makedirs(save_result_dir, exist_ok=True)

# Function to load dataset from txt file, y can be real test values (for test set), or dummy values (when doing prediction)
def load_data(file_path):
    sequences = []
    sequences_onehot = []
    values = []
    site_types = []
    site_nums = []
    seq_names = []
    with open(file_path, 'r') as file:
        for line in file:
            sequence, value, site_type, site_num, seq_name = line.strip().split()  # Assuming columns are space-separated
            encoded_sequence = []
            for ch in sequence:
                if ch == 'N':
                    encoded_sequence.append([0, 0, 0, 0]) # follow Beibei's implementation
                else:
                    one_hot = [0] * 4
                    one_hot["ACGT".index(ch)] = 1
                    encoded_sequence.append(one_hot)
            sequences.append(sequence)
            sequences_onehot.append(encoded_sequence)
            values.append(float(value))
            site_types.append(site_type)
            site_nums.append(site_num)
            seq_names.append(seq_name)
    return torch.tensor(sequences_onehot, dtype=torch.float32), torch.tensor(values, dtype=torch.float32), sequences, site_type, site_nums, seq_names


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
def visualize_attention(model, sequence_onehot, sequence, seq_name, device, save_result_dir):
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
    print(attn_weights.cpu().size())
    attn_weights = attn_weights.squeeze(0).cpu().numpy()  # (seq_len, seq_len)
    plt.imshow(attn_weights, cmap="viridis",vmin=0, vmax=0.36)
    plt.colorbar()
    plt.title("Attention Map")
    x = [i for i in range(len(sequence))]
    ticks = [nucleotide for nucleotide in sequence]
    plt.xticks(x,ticks)
    plt.yticks(x,ticks)
    plt.xlabel("Sequence Positions")
    plt.ylabel("Sequence Positions")
    figname = f"{save_result_dir}{sequence}_{seq_name}_attnMap.png"
    print(figname)
    plt.savefig(figname)
    plt.close()

    # save attention weights
    attn_weights_filename = f"{save_result_dir}{sequence}_{seq_name}_attnWeight.txt"
    np.savetxt(attn_weights_filename, attn_weights)
    print(attn_weights_filename)

    # plot Attention logo 
    # Compute column sums
    column_sums = np.sum(attn_weights, axis=0)
    # X positions for the bars
    x = np.arange(len(sequence))
    # Create the bar plot
    plt.figure(figsize=(8, 3))
    plt.bar(x, column_sums, color="royalblue")
    # Set x-ticks to sequence letters
    plt.xticks(x, list(sequence), fontsize=9)
    #plt.yticks([])
    plt.ylim(0, 8)
    plt.ylabel("Positional Attention Weights")
    plt.xlabel("Sequence Position")
    plt.title("Attention Weights")
    figname = f"{save_result_dir}{sequence}_{seq_name}_attnLogo.png"
    print(figname)
    plt.savefig(figname)
    plt.close()


# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model parameters (must match the trained model)

nlayer = Nlayer
nhead = Nhead
seq_len = SeqLen

input_dim = 4  # A, C, G, T (one-hot encoded)
max_len = seq_len
d_model = 128
batch_size = 256
learning_rate = 0.001

# Load trained model
model = SimpleTransformer(input_dim, d_model, nhead, nlayer, max_len).to(device)
model.load_state_dict(torch.load(trained_model))
model.eval()
print(f"Using trained model {trained_model}")


### Load sequences for prediction with data loader
# *** MAKE SURE YOU HAVE prediction_seq.txt created
# in this format, space separated with 4 columes below:
# seq,
# value(dummy),
# seq_index(for easy plotting along genomic coord),
# seq_name(for easy naming on plot)

# This is only for the purpose of prediction and getting insights: get predicted values, and visualize attention.
pred_seqs_onehot, _ , pred_seqs, pred_seqs_types, pred_seqs_nums, pred_seqs_names = load_data(pred_data_file)
pred_seqs_onehot = pred_seqs_onehot.to(device)


# Run predictions and save predictions to rsult dir
with torch.no_grad():
    # predictions = model(test_sequences_onehot).squeeze()
    predictions = model(pred_seqs_onehot).squeeze()
# save prediction


def save_predictions(input_file, preds, output_file):
    with open(input_file, "r") as f:
        lines = [line.strip().split() for line in f]
    # Ensure preds length matches the number of rows
    if len(preds) != len(lines):
        raise ValueError("Number of predictions does not match the number of input rows.")
    # Append predictions to each row
    updated_lines = [" ".join(line + [str(pred)]) for line, pred in zip(lines, preds)]
    with open(output_file, "w") as f:
        f.write("\n".join(updated_lines))
    print(f"Updated file saved to {output_file}")


# Save predictions values
preds = predictions.cpu().numpy().flatten()
out_file_name = f'{save_result_dir}{pred_data_name}.txt'
save_predictions(pred_data_file, preds, out_file_name)


# Visualize attention for all or some predicted sequences
for i in range(len(pred_seqs)):
    visualize_predseq_onehot = pred_seqs_onehot[i].unsqueeze(0)
    visualize_predseq = pred_seqs[i]
    visualize_predseq_name = pred_seqs_names[i]
    visualize_attention(model, visualize_predseq_onehot, visualize_predseq, visualize_predseq_name, device, save_result_dir)
