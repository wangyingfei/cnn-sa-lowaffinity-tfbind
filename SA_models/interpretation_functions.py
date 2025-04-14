import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import h5py
import sys
import os
import shap


if len(sys.argv) > 9:
    TFname = sys.argv[1]
    ModelType = sys.argv[2]
    SeqLen = int(sys.argv[3])
    DateMark = sys.argv[4]
    Nlayer = int(sys.argv[5])
    Nhead = int(sys.argv[6])
    # added LR and BS for testing out different hyper params sets
    LR = float(sys.argv[7])
    BS = int(sys.argv[8])
    pred_seq = sys.argv[9] 
    # used to load data/interpret_seq/{pred_seq}.h5, and save result into {save_result_dir}interpret_seq_{pred_seq}.txt
else:
    print("Please provide a TFname, ModelType, Seqlen ... etc as command-line arguments")
    sys.exit(1)

#trained_model = f'{TFname}_{ModelType}_models_gridsearch_{DateMark}/lr0.001_batch256_nlayer{Nlayer}_nhead{Nhead}_dmodel128.pth'
trained_model = f'{TFname}_{ModelType}_models_gridsearch_{DateMark}/lr{LR}_batch{BS}_nlayer{Nlayer}_nhead{Nhead}_dmodel128.pth'
pred_data_file = f'data/interpret_seq/{pred_seq}.h5'

# create folders for prediction result:
#save_result_dir = f'prediction/{TFname}_{ModelType}_{DateMark}_nlayer{Nlayer}_nhead{Nhead}/'
save_result_dir = f'prediction/{TFname}_{ModelType}_{DateMark}_nlayer{Nlayer}_nhead{Nhead}_lr{LR}_batch{BS}/'
os.makedirs(save_result_dir, exist_ok=True)



## load h5, for interpreting the sequences
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


## SA model class and hyperparams
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

# load data
test_sequences_onehot, test_values, test_sequences = load_h5_data(pred_data_file)
test_sequences_onehot, test_values = test_sequences_onehot.to(device), test_values.to(device)

# PREVIOUSLY: make predictions using SA model, now this part is integrated into function: calculate_mutation_for_each_window()
#with torch.no_grad():
#    predictions = model(test_sequences_onehot).squeeze()



## this function is NOT used, only copied over to help understand the workflow
def interpret_CNN(outdir, method, interpret_file, tf, param_list, onehot):
    '''
    np.random.seed(1)
    K.set_learning_phase(0)    
    params = [line.rstrip().split('\t')[0].split(',') for line in open(param_list).readlines()]
    
    for i, param in enumerate(params):
        lr_backprop = np.float(param[2])
    
        print ("Start loading model..." + "for parameter " + str(i) + ", "+ ",".join(param) + "\n")
        model = load_model(outdir + "/model_"+ ",".join(param) + ".json", outdir + "/model_" + ",".join(param) + ".h5")
  
        opt = Adam(lr=lr_backprop)
        model.compile(optimizer=opt, loss="mse")

        if method == "GradientTimesInput":
            gradients = GradientTimesInput(model, onehot)
        elif method == "DeconvNet":
            gradients = DeconvNet(model, onehot)
        elif method == "ISM":
            gradients = ISM(model, onehot)
        else:
            raise NameError('Method names not considered in this study!')

        #one example of filename_prefix could be /home/bxin/Documents/low_affinity/multi-module_CNN/out/SELEX_RCmodel/Scr/ISM/
        filename_prefix = outdir + "/" + method + "/" + interpret_file.split('/')[-1][:-3] + "_" + ",".join(param)
        header = np.array(['A', 'C', 'G','T'], dtype='|S32').reshape(1,4)
        np.savetxt(filename_prefix + '.txt', np.vstack((header, gradients)), delimiter = '\t',fmt='%s')
        
        #Here the script assumes that GradientTimesInput and DeconvNet only take one sequence in .h5,
        #whereas ISM can take either one or multiple sequences for long DNA regions. Nevertheness, every time only one 
        #.png file is output
        visualize(filename_prefix, gradients, 80, title="")
    '''

## this function is NOT used in here, just copied over for reference
## the seq2logo step is run separately using the 
def visualize(filename_prefix, matrix, width, title=''):
    '''
    #filename_prefix should be an absolute path of a file.  
    #for short seq, height = 250
    #for long seq, like svb enhancer, height =  
    width = 20*matrix.shape[0]+width
    cmd = 'seq2logo -f ' + filename_prefix + '.txt -o ' +  filename_prefix + '.png -I 5 --colors \'FF0000:T,0000FF:C,FFA500:G,32CD32:A\'' \
    + ' -p ' + str(width) + 'x150 -u \'' + title + '\' -H \'xaxis,fineprint,ends\''
    os.system(cmd)'
    '''

## remain unchanged
def pwm_centralize(pwm):
    '''pwm is of size length X 4'''
    row_mean = pwm.mean(axis=1)
    return pwm-row_mean[:, np.newaxis]

## this function is not used in here, copied over for reference
## modified this function into SA_calculate_mutation_for_each_window(), and used in calculate_mutation_for_long_seq()
def calculate_mutation_for_each_window(model, onehot_i):
    l, c = onehot_i.shape
    mut_pred = np.zeros((l,c))
    for i in range(l):
        for j in range(c):
           temp = np.copy(onehot_i)
           temp[i,:] = 0
           temp[i,j] = 1
           mut_pred[i,j] = model.predict(temp.reshape(1,l,c))[0,0]
    # use the method DeepBind uses in supplemental note 10.1
    ref = np.multiply(onehot_i, mut_pred)
    temp_ref = ref[np.nonzero(ref)]
    ref = np.tile(temp_ref, c).reshape(c,l).T
    
    #for every i,j pair, find the largest predicted_y among (ref, mutaiton and 0)
#    temp_emphasize = np.maximum(ref, mut_pred)
#    emphasize = np.maximum(temp_emphasize, np.zeros((l,c)))   
#    return np.multiply(emphasize, (mut_pred-ref))   #the final output size is l X c
    return np.divide(mut_pred, ref)   #so that mutations will have ratio between (0,1)

# KEY: FOR ISM, modified from the origianl calculate_mutation_for_each_window(), to use SA predictions
def SA_calculate_mutation_for_each_window(model, onehot_i):
    
    onehot_i = onehot_i.cpu().numpy() if isinstance(onehot_i, torch.Tensor) else onehot_i  # Ensure NumPy format

    l, c = onehot_i.shape
    mut_pred = np.zeros((l,c))
    for i in range(l):
        for j in range(c):
            temp = np.copy(onehot_i)
            temp[i,:] = 0
            temp[i,j] = 1
            
            # mut_pred[i,j] = model.predict(temp.reshape(1,l,c))[0,0]
            temp_tensor = torch.tensor(temp, dtype=torch.float32).unsqueeze(0).to(device)  # Shape (1, L, C)
            with torch.no_grad():
                prediction = model(temp_tensor).squeeze().cpu().numpy()
            mut_pred[i, j] = prediction.item()  # Convert single value to scalar
    
    ref = np.multiply(onehot_i, mut_pred)
    temp_ref = ref[np.nonzero(ref)]
    ref = np.tile(temp_ref, c).reshape(c,l).T
    result = np.divide(mut_pred, ref)
    return result

## only changed calculate_mutation_from_each_window() into the updated one for SA models: SA_calculate_mutation_from_each_window()
def calculate_mutation_for_long_seq(model, onehot):   
    '''calculate the mutation map for short seqs from a long seq'''
    temp_n, l, c = onehot.shape  #suppose long seqs have n nucleotide, then temp_n=n-l+1
    n = temp_n + l -1    
    result = np.zeros((n,c))
    
    for i in range(temp_n):  #iterate over all subseqs
        # result[i:i+l,:] += calculate_mutation_for_each_window(model, onehot[i,:,:])
        result[i:i+l,:] += SA_calculate_mutation_for_each_window(model, onehot[i,:,:])

    if n > 2*(l-1):
        denominator_temp = np.append(np.arange(l)[1:], np.repeat(l,(n-2*(l-1))))
        denominator = np.append(denominator_temp, np.flip(np.arange(l),0)[:-1]) 
    else:
        max_cov = n-l+1
        denominator_temp = np.append(np.arange(n-l+1)[1:], np.repeat(max_cov,(n-2*(n-l))))
        denominator = np.append(denominator_temp, np.flip(np.arange(n-l+1),0)[:-1])
        
    if len(denominator) != result.shape[0]:
        print("In calculate_mutation_for_long_seq, dimsional for last step does not match!!!\n")
        
    return np.divide(result, denominator.reshape(n,1))

## remain unchanged
def ISM(model, onehot):
    '''onehot is a .h5 file contains 1-bp shifted kmers '''
    '''Different from DeconvNet and GradientTimesInput, ISM here was designed to output the mutation map for a long sequence'''
    '''get the mutation map for both forward strand and reverse complement strand'''
    fwd_mutation_map = calculate_mutation_for_long_seq(model, onehot)   #size is len_of_long_seq X 4
#    rev_mutation_map = calculate_mutation_for_long_seq(model, onehot[:, ::-1, ::-1])
    
    ##save mutation maps for mutation logos
    #first get the original long seqs:
    temp_n, l, c = onehot.shape
    n = temp_n + l -1
    long_seq = np.zeros((n,c))
    for i in range(temp_n):
        # long_seq[i:i+l,:] += onehot[i,:,:]
        long_seq[i:i+l,:] = np.add(long_seq[i:i+l,:], onehot[i,:,:])
    #long_seq is of size len_of_long_seq X 4
    long_seq[long_seq>0] = 1
    
    #generate the importance of every position by adding the negative values, get a (len_of_long_seq, 1) vector
    #temp_fwd_mutation = np.copy(fwd_mutation_map)
    #temp_fwd_mutation[temp_fwd_mutation>0] = 0
    #temp_fwd_mutation_aggreg = -np.sum(temp_fwd_mutation, axis=1).reshape(n,1)
    
    temp_fwd_mutation = np.copy(fwd_mutation_map)
    '''mainly take care of those 0 elements in the matrix'''
    temp_fwd_mutation[temp_fwd_mutation<=0] = 1e-5
    '''take a log first, then sum up those that have >0 result, meaning ref > mut'''
    temp_fwd_mutation = np.log(temp_fwd_mutation) 
    centralized_temp_fwd_mutation = pwm_centralize(temp_fwd_mutation)
    '''visualize only those appear in the sequence, with original height'''
    visualizeD = np.multiply(long_seq, centralized_temp_fwd_mutation)
    return visualizeD

###############################################################

#GTI method, implemented a new function for SA model
def GradientTimesInput(model, onehot):
    """
    Compute Gradient × Input interpretation for a transformer model.

    Parameters:
    - model: PyTorch model
    - onehot: One-hot encoded input (numpy array of shape (1, L, C))

    Returns:
    - visualizeD: Gradient × Input interpretation (numpy array)
    """
    # Convert input to a PyTorch tensor and ensure it requires gradients
    onehot_tensor = torch.tensor(onehot, dtype=torch.float32, requires_grad=True).to(device)

    # Forward pass
    model.eval()
    output = model(onehot_tensor)  # Get model prediction

    # Compute gradients (assumes scalar output; adjust if necessary)
    grad = torch.autograd.grad(outputs=output, inputs=onehot_tensor, 
                               grad_outputs=torch.ones_like(output),
                               create_graph=False, retain_graph=False)[0]

    # Compute Gradient × Input
    visualizeD = (grad * onehot_tensor).detach().cpu().numpy().squeeze(0)  # Removes batch dimension

    return visualizeD

###############################################################

# SHAP 
# DeepSHAP integrates SHAP with DeepLIFT, ideal for ReLU based activation
# my simpleTransformer uses sigmoid activation, thus GradientSHAP is used
# SHAP requires a baseline input (e.g., all-zeros sequence)
def GradientSHAP(model,onehot,baseline):
    # Convert baseline to a PyTorch tensor if it's a NumPy array
    if isinstance(baseline, np.ndarray):
        baseline_tensor = torch.tensor(baseline, dtype=torch.float32).to(device)
    else:
        baseline_tensor = baseline.to(device)  # Assume it's already a torch tensor
    # Initialize Deep SHAP explainer
    explainer = shap.GradientExplainer(model, baseline_tensor)
    shap_values = explainer.shap_values(onehot)[0]  # Remove singleton list dimension
    # Remove extra dimensions if necessary
    return shap_values.squeeze()  # Shape should be (14,4)

###############################################################

# TODO: SA SHAP (SHAP for Self attention based model)
def SA_SHAP(model, onehot_input, baseline):
    """
    Compute Self-Attention SHAP (SA-SHAP) values for a transformer model.
    
    Args:
        model: Transformer model
        onehot_input: One-hot encoded input sequence (batch_size, seq_length, num_features)
        baseline: Baseline input for SHAP comparisons (same shape as onehot_input)
        device: Device to run computation on ("cuda" or "cpu")
    
    Returns:
        shap_values: SHAP values for self-attention importance
    """
    model.eval()
    onehot_input = torch.tensor(onehot_input, dtype=torch.float32).to(device)
    baseline = torch.tensor(baseline, dtype=torch.float32).to(device)

    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super(ModelWrapper, self).__init__()
            self.model = model  # Store the original model

        def forward(self, x):
            """
            Custom forward method that extracts attention weights.
            """
            with torch.no_grad():
                linear_in = self.model.linear_in
                positional_encoding = self.model.positional_encoding

                lineared = linear_in(x)
                positioned = positional_encoding(lineared)

                all_attn_weights = []
                for i in range(len(self.model.transformer.layers)):
                    attn_layer = self.model.transformer.layers[i].self_attn
                    _, attn_weights = attn_layer(
                        query=positioned, key=positioned, value=positioned, need_weights=True
                    )
                    all_attn_weights.append(attn_weights.mean(dim=1))  # Average over heads

            return torch.stack(all_attn_weights, dim=0)  # Shape: (num_layers, batch_size, seq_len, seq_len)

    wrapped_model = ModelWrapper(model).to(device)

    # Initialize SHAP GradientExplainer (works with PyTorch models)
    explainer = shap.GradientExplainer(wrapped_model, baseline)

    # Compute SHAP values for input self-attention scores
    shap_values = explainer.shap_values(onehot_input)

    return shap_values

################################################################

# (optinal) KEY: new interpretaion method IG
def integrated_gradients(model, onehot_seq, baseline=None, steps=50, device='cuda'):
    """
    Computes Integrated Gradients for a given model and one-hot encoded DNA sequence.
    
    Parameters:
        model: Transformer model
        onehot_seq: One-hot encoded DNA sequence (NumPy array of shape [L, C])
        baseline: Baseline input (default: all zeros of shape [L, C])
        steps: Number of integration steps
        device: Device to run computations on ('cuda' or 'cpu')
    
    Returns:
        IG attributions of shape [L, C] as a NumPy array
    """
    onehot_seq = torch.tensor(onehot_seq, dtype=torch.float32).unsqueeze(0).to(device)  # Shape: [1, L, C]

    if baseline is None:
        baseline = torch.zeros_like(onehot_seq)  # Default: all-zero baseline
    else:
        baseline = torch.tensor(baseline, dtype=torch.float32).unsqueeze(0).to(device)

    # Create interpolation steps between baseline and input
    interpolated_inputs = [
        baseline + (float(i) / steps) * (onehot_seq - baseline) 
        for i in range(steps + 1)
    ]
    
    interpolated_inputs = torch.stack(interpolated_inputs).to(device)  # Shape: (steps+1, 1, L, C)

    # Enable gradient tracking
    interpolated_inputs.requires_grad_()
    
    # Forward pass and compute gradients
    model.eval()
    total_gradients = torch.zeros_like(onehot_seq, device=device)
    
    for i in range(steps):
        interpolated_inputs[i].requires_grad_()
        pred = model(interpolated_inputs[i].unsqueeze(0)).squeeze()  # Forward pass
        pred.backward(retain_graph=True)  # Backpropagate
        total_gradients += interpolated_inputs[i].grad  # Sum gradients

    # Compute IG
    avg_gradients = total_gradients / steps
    integrated_gradients = (onehot_seq - baseline) * avg_gradients

    return integrated_gradients.squeeze().cpu().numpy()  # Convert back to NumPy array



### main work flow:
'''
0. modify the inner most function for it to take SA model prediction, then run the following steps:
1. gradients = ISM(model, onehot), to get the predicted importance matrix , or other methods
2. header = np.array(['A', 'C', 'G','T'], dtype='|S32').reshape(1,4)
   np.savetxt(filename_prefix + '.txt', np.vstack((header, gradients)), delimiter = '\t',fmt='%s')
   to save the matrix into a format required by seq2logo
3. visualize(filename_prefix, gradients, 80, title="")
   specifically, run seq2logo in command line separately using
   seq2logo -f filename_prefix.txt -o filename_prefix.png -I 5 --colors 'FF0000:T,0000FF:C,FFA500:G,32CD32:A' -p widthx150 -u 'title' -H 'xaxis,fineprint,ends'
'''

header = np.array(['A', 'C', 'G','T']).reshape(1,4)

## ISM
gradients = ISM(model, test_sequences_onehot)
save_result_file = f'{save_result_dir}interpret_seq_{pred_seq}_ISM.txt'
np.savetxt(save_result_file, np.vstack((header, gradients)), delimiter = '\t',fmt='%s')
print(f"interpretation matrix saved to {save_result_file}")

## GTI
#gradients = GradientTimesInput(model, test_sequences_onehot)
#save_result_file = f'{save_result_dir}interpret_seq_{pred_seq}_GTI.txt'
#np.savetxt(save_result_file, np.vstack((header, gradients)), delimiter = '\t',fmt='%s')
#print(f"interpretation matrix saved to {save_result_file}")

## SHAP
# use all zeros as the baseline
baseline1 = np.zeros((1, 14, 4))
gradients = GradientSHAP(model,test_sequences_onehot,baseline1)
save_result_file = f'{save_result_dir}interpret_seq_{pred_seq}_GradientSHAP_zeros.txt'
np.savetxt(save_result_file, np.vstack((header, gradients)), delimiter = '\t',fmt='%s')
print(f"interpretation matrix saved to {save_result_file}")
# using the entire training set as the baseline
train_sequences_onehot, train_values, train_sequences = load_h5_data('data/SELEX_canonical/Ubx/Ubx_train.h5')
train_sequences_onehot, train_values = train_sequences_onehot.to(device), train_values.to(device)
baseline2 = train_sequences_onehot
gradients = GradientSHAP(model,test_sequences_onehot,baseline2)
save_result_file = f'{save_result_dir}interpret_seq_{pred_seq}_GradientSHAP_trainset.txt'
np.savetxt(save_result_file, np.vstack((header, gradients)), delimiter = '\t',fmt='%s')
print(f"interpretation matrix saved to {save_result_file}")


##TODO
##baseline = np.zeros((1, 14, 4))
##gradients = SA_SHAP(model, test_sequences_onehot, baseline)
##save_result_file = f'{save_result_dir}interpret_seq_{pred_seq}_SASHAP.txt'


