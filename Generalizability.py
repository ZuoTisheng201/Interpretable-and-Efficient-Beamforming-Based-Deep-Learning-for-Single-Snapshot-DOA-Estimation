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

s_minus_results = scipy.io.loadmat('Generalizability_s_minus_results.mat')
s_minus1 = s_minus_results['s_minus_results']
s_minus_real1 = np.real(s_minus1)
s_minus_imag1 = np.imag(s_minus1)

mat_data1 = scipy.io.loadmat('Generalizability_beamforming_dataset.mat')
beamforming_vectors = mat_data1['beamforming_vectors']
labels = mat_data1['labels']
SNR = mat_data1['SNRs']
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
    if len(peaks) >= 4:
        top_peaks = peaks[np.argsort(spectrum[peaks])[-4:]]
    else:
        top_peaks = peaks
    estimated_doas_indices.append(top_peaks)

angle_range = np.linspace(-90, 90, spectrum_estimates.shape[1])
estimated_doas = np.array([angle_range[indices] for indices in estimated_doas_indices])

SNRs = np.unique(SNR.flatten())  # 确保 SNR 是一个具有唯一值的平面数组

# 提到的固定 DOA 值
fixed_doas = np.array([-62.2, -21.9, 5.3, 45.1])

# 初始化一个列表以存储命中率
hit_rates = []

# 计算每个 SNR 下的命中率
for snr in SNRs:
    mask = SNR.flatten() == snr
    current_estimated_doas = estimated_doas[mask]

    hits = 0
    for est_doas in current_estimated_doas:
        if len(est_doas) == 4:
            # Check if the estimated DOA is within ±1 degree of any fixed DOA
            hit = np.any([np.abs(fixed_doa - est_doa) <= 2 for fixed_doa in fixed_doas for est_doa in est_doas])
            hits += hit

    hit_rate = hits / len(current_estimated_doas) if len(current_estimated_doas) > 0 else 0
    hit_rates.append(hit_rate)

snr_levels = np.array(SNRs)
hit_rates = np.array(hit_rates)

plt.figure(figsize=(10, 6))
plt.plot(snr_levels, hit_rates, marker='o', linestyle='-')
plt.xlabel('SNR (dB)')
plt.ylim(0, 1)
plt.ylabel('Hit Rate')
plt.title('Generalizability')
plt.grid(True)
plt.show()
