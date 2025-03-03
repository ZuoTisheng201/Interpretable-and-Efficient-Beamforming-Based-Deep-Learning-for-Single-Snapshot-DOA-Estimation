import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.metrics import mean_squared_error
from model import DeepMPDRModel
import scipy.io

best_model_path = "best_DeepMPDRModel.pth"
model = DeepMPDRModel()
model.load_state_dict(torch.load(best_model_path))
model.eval()

# 加载训练数据集
s_minus_results = scipy.io.loadmat('accuracy_s_minus_results2.mat')
s_minus1 = s_minus_results['s_minus_results']
s_minus_real1 = np.real(s_minus1)
s_minus_imag1 = np.imag(s_minus1)

mat_data1 = scipy.io.loadmat('accuracy_beamforming_dataset2.mat')
beamforming_vectors = mat_data1['beamforming_vectors']
labels = mat_data1['labels']
DoA = mat_data1['DoAs']
SNR = mat_data1['SNRs']
beamforming_vectors_real1 = np.real(beamforming_vectors)
beamforming_vectors_imag1 = np.imag(beamforming_vectors)

s_minus_train = np.concatenate([s_minus_real1, s_minus_imag1], axis=-1)
bf_vectors_train = np.concatenate([beamforming_vectors_real1, beamforming_vectors_imag1], axis=-1)

# 将输入转换为torch张量并适当重塑
s_minus_tensor = torch.tensor(s_minus_train, dtype=torch.float32)
bf_vector_tensor = torch.tensor(bf_vectors_train, dtype=torch.float32)
num_instances = s_minus_tensor.shape[0]
# spectrum_estimates = []

spectrum_estimates = model(s_minus_tensor, bf_vector_tensor).detach().numpy()

all_peaks_indices = []
estimated_doas = []
rmse_values = []

for i in range(spectrum_estimates.shape[0]):
    spectrum = spectrum_estimates[i, :]
    peaks, properties = find_peaks(spectrum, height=0.1)

    angle_range = np.linspace(-90, 90, spectrum_estimates.shape[1])

    peak_idx = peaks[np.argmax(properties['peak_heights'])]  # 选择最高峰的索引
    estimated_doa = angle_range[peak_idx]
    estimated_doas.append(estimated_doa)

estimated_doas = np.array(estimated_doas)
true_doas = DoA.flatten()

num_data_per_snr = 300
snr_range = np.arange(-10, 21, 4)
rmse_values = []

# 遍历每个SNR值，计算对应的RMSE
for i, snr in enumerate(snr_range):
    start_idx = i * num_data_per_snr
    end_idx = start_idx + num_data_per_snr
    # 提取对应SNR值的真实DOA和估计DOA
    true_doas_snr = true_doas[start_idx:end_idx]
    estimated_doas_snr = estimated_doas[start_idx:end_idx]

    rmse = np.sqrt(mean_squared_error(true_doas_snr, estimated_doas_snr))

    rmse_values.append(rmse)

# 绘制SNR与RMSE的曲线图
plt.figure(figsize=(10, 6))
plt.plot(snr_range, rmse_values, marker='o', linestyle='-')
plt.xlabel('SNR (dB)')
plt.ylabel('RMSE (degrees)')
plt.title('Accuracy')
plt.grid(True)
plt.show()