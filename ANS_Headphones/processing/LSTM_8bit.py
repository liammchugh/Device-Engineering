import torch
import copy
import io
import numpy as np
from torch.quantization import default_qconfig, quantize_dynamic

class LSTMDynamicModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTMDynamicModel, self).__init__()
        self.qconfig = default_qconfig
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers=num_layers).to(dtype=torch.float)

    def forward(self, x, hiddens):
        x, hiddens = self.lstm(x, hiddens)
        return x, hiddens

# Parameters
input_size = 250
hidden_size = 150
num_layers = 2
niter = 100
sequence_length = 250

model = LSTMDynamicModel(input_size, hidden_size, num_layers).eval()

# Manually setting weight values for testing
vals = [[100, -155], [100, -155], [-155, 100], [-155, 100], [100, -155], [-155, 100], [-155, 100], [100, -155]]

# Number of gates in LSTM (input, forget, cell, output)
num_gates = 4

# Ensure the model is an LSTM
if isinstance(model.lstm, torch.nn.LSTM):
    # Calculate the required weight shapes
    input_weights_shape = (num_gates * hidden_size, input_size)
    hidden_weights_shape = (num_gates * hidden_size, hidden_size)
    
    # Flatten and repeat `vals` to match the required number of elements for input-to-hidden and hidden-to-hidden weights
    vals_flat = np.array(vals).flatten()  # Convert to a 1D array
    
    # Ensure there are enough values to fill the weight matrices
    input_vals_needed = np.prod(input_weights_shape)
    hidden_vals_needed = np.prod(hidden_weights_shape)
    
    # Repeat vals_flat to have enough values for the weight matrices
    repeated_vals_ih = np.tile(vals_flat, input_vals_needed // len(vals_flat) + 1)[:input_vals_needed]
    repeated_vals_hh = np.tile(vals_flat, hidden_vals_needed // len(vals_flat) + 1)[:hidden_vals_needed]
    
    # Initialize the input-to-hidden and hidden-to-hidden weights with the repeated values
    model.lstm.weight_ih_l0 = torch.nn.Parameter(torch.tensor(repeated_vals_ih, dtype=torch.float).view(input_weights_shape), requires_grad=False)
    model.lstm.weight_hh_l0 = torch.nn.Parameter(torch.tensor(repeated_vals_hh, dtype=torch.float).view(hidden_weights_shape), requires_grad=False)
    
    # Initialize biases to zeros (4 * hidden_size to account for each gate)
    model.lstm.bias_ih_l0 = torch.nn.Parameter(torch.zeros(num_gates * hidden_size), requires_grad=False)
    model.lstm.bias_hh_l0 = torch.nn.Parameter(torch.zeros(num_gates * hidden_size), requires_grad=False)
    
# Reference model for comparison
ref = copy.deepcopy(model.lstm)

# Apply quantization/precision
model_int8 = quantize_dynamic(model=model, dtype=torch.qint8)
model_fp16 = model.half()

cell_int8 = model_int8.lstm
cell_fp16 = model_fp16.lstm

assert isinstance(cell_int8, torch.nn.quantized.dynamic.LSTM), "LSTM quantization failed for int8"

# Random test input of shape [niter, sequence_length, input_size]
x = torch.randn(niter, sequence_length, input_size, dtype=torch.float)

# Hidden and cell states for hidden_size=500
hx = torch.randn(num_layers, sequence_length, hidden_size, dtype=torch.float)
cx = torch.randn(num_layers, sequence_length, hidden_size, dtype=torch.float)

x_fp16 = x.to(torch.float16)  # Convert inputs to FP16
hx_fp16 = hx.to(torch.float16)
cx_fp16 = cx.to(torch.float16)

# Hidden state tuple
hiddens = (hx, cx)
# Compare quantized model with reference
ref_out, ref_hid = ref(x, hiddens)
output_int8, final_hiddens_int8 = cell_int8(x, hiddens)
output_fp16, final_hiddens_fp16 = cell_fp16(x_fp16, (hx_fp16, cx_fp16))

# Adjust the tolerance for the comparison
# torch.testing.assert_close(output_int8, ref_out, rtol=1e-2, atol=1e-3)

# Compare hidden states with relaxed tolerance
# for out_val, ref_val in zip(final_hiddens_int8, ref_hid):
#     torch.testing.assert_close(out_val, ref_val, rtol=1e-2, atol=1e-3)

# TorchScript tracing and saving
class ScriptWrapper(torch.nn.Module):
    def __init__(self, cell):
        super(ScriptWrapper, self).__init__()
        self.cell = cell

    def forward(self, x, hiddens):
        return self.cell(x, hiddens)

# Trace and save the model
cell_script = torch.jit.trace(ScriptWrapper(cell_int8), (x, (hx, cx)))
torch.jit.save(cell_script, 'quantized_lstm_model.pt')
cell_script16 = torch.jit.trace(ScriptWrapper(cell_fp16), (x_fp16, (hx_fp16, cx_fp16)))
torch.jit.save(cell_script16, 'fp16_lstm_model.pt')


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and run the saved model FP16
prec = "fp16"
if prec == "fp16":
    loaded_model = torch.jit.load('fp16_lstm_model.pt').to(device)
    xt = x_fp16.to(device)
    hx_fp16 = hx_fp16.to(device)
    cx_fp16 = cx_fp16.to(device)
    hiddenst = (hx_fp16, cx_fp16)
else:
    loaded_model = torch.jit.load('quantized_lstm_model.pt').to(device)
    xt = x.to(device)
    hx = hx.to(device)
    cx = cx.to(device)
    hiddenst = (hx, cx)

loaded_output, loaded_hiddens = loaded_model(xt, hiddenst)

# Verify output after loading
# torch.testing.assert_close(loaded_output, ref_out)

import time

# Timing float32 model
start = time.time()
for _ in range(100):  # Run inference multiple times to get an average
    output_float = ref(x, hiddens)
end = time.time()
float_time = end - start

# Timing quantized int8 model
start = time.time()
for _ in range(100):
    output_qaunt = loaded_model(xt, hiddenst)
end = time.time()
quant_time = end - start

print(f"Float32 Inference Time: {float_time}")
print(f"{prec} Inference Time: {quant_time}")
print(f"Speedup: {float_time / quant_time}")


import torch.autograd.profiler as profiler

# Profile the quantized model
with profiler.profile(record_shapes=True) as prof_quant:
    loaded_model(xt, hiddenst)
quant_prof = prof_quant.key_averages().table(sort_by="cpu_time_total")

# Profile the reference model
with profiler.profile(record_shapes=True) as prof_ref:
    ref(x, hiddens)
ref_prof = prof_ref.key_averages().table(sort_by="cpu_time_total")

# Extract the max CPU time process for both models
quant_max_cpu_time = max(prof_quant.key_averages(), key=lambda evt: evt.cpu_time_total)
ref_max_cpu_time = max(prof_ref.key_averages(), key=lambda evt: evt.cpu_time_total)

print(f"{prec} Model Max CPU Time Process:")
print(quant_max_cpu_time)

print("\nReference Model Max CPU Time Process:")
print(ref_max_cpu_time)
