import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pdb
import h5py
import time
import sys
import os

if len(sys.argv) > 4:
    TFname = sys.argv[1]
    ModelType = sys.argv[2]
    SeqLen = sys.argv[3]
    DateMark = sys.argv[4]
else:
    print("Please provide a TFname and a Seqlen as a command-line arguments")
    sys.exit(1)

# create folders for result:
save_models_dir = f'{TFname}_{ModelType}_models_gridsearch_{DateMark}'
save_attnMap_dir = f'{TFname}_{ModelType}_attnMap_gridsearch_{DateMark}'
save_lossCurve_dir = f'{TFname}_{ModelType}_lossCurve_gridsearch_{DateMark}'
os.makedirs(save_models_dir, exist_ok=True)
os.makedirs(save_attnMap_dir, exist_ok=True)
os.makedirs(save_lossCurve_dir, exist_ok=True)


# Define dataset class
class DNADataset(Dataset):
    def __init__(self, sequences, values):
        self.sequences = sequences
        self.values = values

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.values[idx]


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

def evaluate_model(model, data_loader, data_name, device, verbose):
    model.eval()
    all_predictions = []
    all_targets = []
    loss = 0

    with torch.no_grad():
        for batch in data_loader:
            sequences, targets = batch
            sequences, targets = sequences.to(device), targets.to(device)

            outputs = model(sequences).squeeze()  # (batch_size,)
            loss += criterion(outputs, targets).item()

            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

        # Compute Pearson's correlation coefficient
        _, _, r_value, _, _ = stats.linregress(all_predictions, all_targets)
        pearson_corr, _ = stats.pearsonr(all_predictions, all_targets)
        loss /= len(data_loader)
        if verbose == True:
            print(f"Loss on {data_name}: {loss:.4f}, Pearson Correlation: {pearson_corr:.4f} , R2: {r_value**2:.4f}")

    return loss, pearson_corr, r_value**2




def plot_loss(train_losses, test_losses, figpath):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", marker='o')
    plt.plot(range(1, len(test_losses) + 1), test_losses, label="Test Loss", marker='s')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Testing Loss Curve")
    plt.legend()
    plt.grid()
    #plt.savefig(f'Ubx_lossCurve_gridsearch_0219/LossCurve_{postfix}.png')
    #plt.savefig(f'Ubx_double_lossCurve_gridsearch_0221/LossCurve_{postfix}.png')
    #plt.savefig(f'Ubx_RCaug_lossCurve_gridsearch_0223/LossCurve_{postfix}.png')
    plt.savefig(figpath)
    plt.close()

def plot_R2(test_R2s, figpath):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(test_R2s) + 1), test_R2s, label="Test R2", marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("R2")
    plt.title("Testing R2 Curve")
    plt.legend()
    plt.grid()
    #plt.savefig(f'Ubx_lossCurve_gridsearch_0219/LossCurve_{postfix}_R2.png')
    #plt.savefig(f'Ubx_double_lossCurve_gridsearch_0221/LossCurve_{postfix}_R2.png')
    #plt.savefig(f'Ubx_RCaug_lossCurve_gridsearch_0223/LossCurve_{postfix}_R2.png')
    plt.savefig(figpath)
    plt.close()
    

# Visualization function for attention map
def visualize_attention(model, sequence_onehot, sequence, device, figpath):
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
    #figname = 'Ubx_attnMap_gridsearch_0219/'+ sequence + f'_{postfix}.png'
    #figname = 'Ubx_double_attnMap_gridsearch_0221/'+ sequence + f'_{postfix}.png'
    #figname = 'Ubx_RCaug_attnMap_gridsearch_0223/'+ sequence + f'_{postfix}.png'
    plt.savefig(figpath)
    plt.close()

# Hyperparameters
input_dim = 4  # A, C, G, T (one-hot encoded)
seq_len = int(SeqLen)
max_len = seq_len
epochs = 120


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# load dataset from h5
#train_sequences_onehot, train_values, train_sequences = load_h5_data("data/SELEX_double/Ubx/Ubx_train.h5")
#test_sequences_onehot, test_values, test_sequences = load_h5_data("data/SELEX_double/Ubx/Ubx_test.h5")

#train_sequences_onehot, train_values, train_sequences = load_h5_data("data/SELEX_double/Ubx/Ubx_train.h5")
#test_sequences_onehot, test_values, test_sequences = load_h5_data("data/SELEX_double/Ubx/Ubx_test.h5")

#train_sequences_onehot, train_values, train_sequences = load_h5_data("data/SELEX_RCaugmented/Ubx/Ubx_train.h5")
#test_sequences_onehot, test_values, test_sequences = load_h5_data("data/SELEX_RCaugmented/Ubx/Ubx_test.h5")

train_h5_path = f'data/SELEX_{ModelType}/{TFname}/{TFname}_train.h5'
test_h5_path = f'data/SELEX_{ModelType}/{TFname}/{TFname}_test.h5'
train_sequences_onehot, train_values, train_sequences = load_h5_data(train_h5_path)
test_sequences_onehot, test_values, test_sequences = load_h5_data(test_h5_path)
# pdb.set_trace()

# Prepare DataLoaders
train_dataset = DNADataset(train_sequences_onehot, train_values)
test_dataset = DNADataset(test_sequences_onehot, test_values)


#for nlayer in [1]:
#    for nhead in [1]:
for nlayer in [2]: # [1,2,4]:
    for nhead in [8]: #[1,2,4,8]:
        for d_model in [128]:
            for batch_size in [256,512,1024]:
                for learning_rate in [1e-2,1e-3,1e-4]:
                    
                    postfix = f"lr{learning_rate}_batch{batch_size}_nlayer{nlayer}_nhead{nhead}_dmodel{d_model}"
                    print(f"===================={postfix}================")
                    
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

                    # Model, loss, optimizer
                    model = SimpleTransformer(input_dim, d_model, nhead, nlayer, max_len).to(device)
                    criterion = nn.MSELoss()
                    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

                    # Training loop
                    train_losses = []
                    # train_losses_from_function = [] # test if the function give the same loss
                    test_losses = []
                    test_R2s = []

                    for epoch in range(epochs):
                        model.train()
                        train_loss = 0

                        for batch in train_loader:
                            sequences, targets = batch
                            sequences, targets = sequences.to(device), targets.to(device)

                            optimizer.zero_grad()
                            outputs = model(sequences)  # (batch_size, 1)
                            loss = criterion(outputs.squeeze(), targets)  # MSE loss
                            loss.backward()
                            optimizer.step()
                            train_loss += loss.item()

                        train_loss /= len(train_loader)
                        train_losses.append(np.log10(train_loss))
                        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}")

                        # test, what if i use the evaluate model function - why arent the loss the same?? otherwise how do I calculate R2 for training set
                        # train_loss_from_function, _ , _ = evaluate_model(model, train_loader, 'train data', device, verbose=True)
                        # train_losses_from_function.append(train_loss_from_function)
                        
                        # check test loss for this epoch, save test loss and test R2
                        test_loss, _ , R2 = evaluate_model(model, test_loader, 'test data', device, verbose=False)
                        test_losses.append(np.log10(test_loss))
                        test_R2s.append(R2)

                    # Check loss plots during training, and evaluate model on test set
                    postfix = f"lr{learning_rate}_batch{batch_size}_nlayer{nlayer}_nhead{nhead}_dmodel{d_model}"
                    loss_figpath = f'{TFname}_{ModelType}_lossCurve_gridsearch_{DateMark}/LossCurve_{postfix}.png'
                    plot_loss(train_losses, test_losses, figpath=loss_figpath)
                    R2_figpath = f'{TFname}_{ModelType}_lossCurve_gridsearch_{DateMark}/LossCurve_{postfix}_R2.png'
                    plot_R2(test_R2s, figpath=R2_figpath)

                    print("Final model evaluation to print test R2 and pearson")
                    evaluate_model(model, test_loader, 'test data', device, verbose=True)

                    # end = time.time()

                    # print(f"Total: {end-start}")

                    ###  Save the trained model
                    #model_name = 'Ubx_models_gridsearch_0219/' + f'{postfix}.pth'
                    #model_name = 'Ubx_double_models_gridsearch_0221/' + f'{postfix}.pth'
                    #model_name = 'Ubx_RCaug_models_gridsearch_0223/' + f'{postfix}.pth'
                    model_name = f'{TFname}_{ModelType}_models_gridsearch_{DateMark}/{postfix}.pth'
                    torch.save(model.state_dict(), model_name)
                    print("Model saved")

                    #print(f"Total: {end-start}")

                    # Attention maps visualization

                    ### USING CANONICAL REDUCED KMER TXT DATA
                    '''
                    # UBX motif: TTAAT(TA)
                    # UBX test [0] ATGATTTATTACCA, 0.868
                    sample_sequence1_onehot = test_sequences_onehot[0].unsqueeze(0).to(device)  # Add batch dim (1, seq_len, vocab_size)
                    sample_sequence1 = test_sequences[0]
                    visualize_attention(model, sample_sequence1_onehot, sample_sequence1, device)
                    '''

                    ### USING h5 data
                    sample_sequence1_onehot = test_sequences_onehot[0].unsqueeze(0).to(device)  # Add batch dim (1, seq_len, vocab_size)
                    sample_sequence1 = test_sequences[0]

                    attn_figpath = f'{TFname}_{ModelType}_attnMap_gridsearch_{DateMark}/{sample_sequence1}_{postfix}.png'
                    visualize_attention(model, sample_sequence1_onehot, sample_sequence1, device, attn_figpath)