import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.signal import find_peaks
from model import DeepMPDRModel
import scipy.io

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepMPDRModel().to(device)
model_path = "best_DeepMPDRModel.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

s_minus_results = scipy.io.loadmat('Separability_s_minus_results4.mat')
s_minus1 = s_minus_results['s_minus_results']
s_minus_real1 = np.real(s_minus1)
s_minus_imag1 = np.imag(s_minus1)

mat_data1 = scipy.io.loadmat('Separability_beamforming_dataset4.mat')
beamforming_vectors = mat_data1['beamforming_vectors']
labels = mat_data1['labels']
DoA = mat_data1['DoAs']
delta_theta = mat_data1['Delta_theta']
beamforming_vectors_real1 = np.real(beamforming_vectors)
beamforming_vectors_imag1 = np.imag(beamforming_vectors)

s_minus_train = np.concatenate([s_minus_real1, s_minus_imag1], axis=-1)
bf_vectors_train = np.concatenate([beamforming_vectors_real1, beamforming_vectors_imag1], axis=-1)

# 将输入转换为torch张量并适当重塑
s_minus_tensor = torch.tensor(s_minus_train, dtype=torch.float32).to(device)
bf_vector_tensor = torch.tensor(bf_vectors_train, dtype=torch.float32).to(device)
num_instances = s_minus_tensor.shape[0]
# spectrum_estimates = []

spectrum_estimates = model(s_minus_tensor, bf_vector_tensor).detach().cpu().numpy()

estimated_doas_indices = []
for spectrum in spectrum_estimates:
    peaks, _ = find_peaks(spectrum)
    top_peaks = peaks[np.argsort(spectrum[peaks])[-2:]] if len(peaks) >= 2 else peaks
    estimated_doas_indices.append(top_peaks)

# 将索引转换为角度
angle_range = np.linspace(-90, 90, spectrum_estimates.shape[1])
estimated_doas = np.array([angle_range[indices] for indices in estimated_doas_indices])

# 初始化命中率统计
hit_rates = []
Delta_theta = np.unique(delta_theta.flatten())

# 计算每个delta_theta下的命中率
for delta in Delta_theta:
    mask = delta_theta.flatten() == delta
    # 创建了一个布尔数组（称为掩码），用于标识delta_theta数组中等于当前循环中考虑的delta值的所有元素。这里delta_theta.flatten()的作用是将delta_theta数组转换成一维数组，以便可以与单个delta值进行比较
    current_doas = DoA[mask]
    current_estimated_doas = estimated_doas[mask]

    hits = 0
    for true_doas, est_doas in zip(current_doas, current_estimated_doas):
        if len(est_doas) == 2:
            hit = np.any(np.abs(true_doas[:, None] - est_doas) <= 3)
            hits += hit

    hit_rate = hits / len(current_doas)
    hit_rates.append(hit_rate)

adjusted_Delta_theta = Delta_theta - 10
# 绘制delta_theta与命中率的图
plt.figure(figsize=(10, 6))
plt.plot(adjusted_Delta_theta, hit_rates, marker='o', linestyle='-')
plt.xlabel('Delta Theta (Degrees)')
plt.ylim(0.5, 1)
plt.ylabel('Hit Rate')
plt.title('Separability')
plt.grid(True)
plt.show()