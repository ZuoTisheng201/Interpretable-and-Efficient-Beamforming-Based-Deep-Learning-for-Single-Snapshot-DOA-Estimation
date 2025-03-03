import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from tqdm import tqdm


# 加载训练数据集
s_minus_results_data1 = scipy.io.loadmat('s_minus_results.mat')
s_minus1 = s_minus_results_data1['s_minus_results']
s_minus_real1 = np.real(s_minus1)
s_minus_imag1 = np.imag(s_minus1)

mat_data1 = scipy.io.loadmat('beamforming_dataset_extended.mat')
beamforming_vectors1 = mat_data1['beamforming_vectors']
labels_train = mat_data1['labels']
beamforming_vectors_real1 = np.real(beamforming_vectors1)
beamforming_vectors_imag1 = np.imag(beamforming_vectors1)

s_minus_train = np.concatenate([s_minus_real1, s_minus_imag1], axis=1)
bf_vectors_train = np.concatenate([beamforming_vectors_real1, beamforming_vectors_imag1], axis=1)

# 加载测试数据集
s_minus_results_data2 = scipy.io.loadmat('s_minus_results2.mat')
s_minus2 = s_minus_results_data2['s_minus_results']
s_minus_real2 = np.real(s_minus2)
s_minus_imag2 = np.imag(s_minus2)

mat_data2 = scipy.io.loadmat('beamforming_dataset_extended2.mat')
beamforming_vectors2 = mat_data2['beamforming_vectors']
labels_test = mat_data2['labels']
# print(beamforming_vectors.shape)
beamforming_vectors_real2 = np.real(beamforming_vectors2)
beamforming_vectors_imag2 = np.imag(beamforming_vectors2)

s_minus_test = np.concatenate([s_minus_real2, s_minus_imag2], axis=1)
bf_vectors_test = np.concatenate([beamforming_vectors_real2, beamforming_vectors_imag2], axis=1)


# 转换为PyTorch张量，这里假设labels是回归目标或需要浮点处理的情况
s_minus_train_tensor = torch.tensor(s_minus_train, dtype=torch.float32)
s_minus_val_tensor = torch.tensor(s_minus_test, dtype=torch.float32)
bf_vectors_train_tensor = torch.tensor(bf_vectors_train, dtype=torch.float32)
bf_vectors_val_tensor = torch.tensor(bf_vectors_test, dtype=torch.float32)
labels_train_tensor = torch.tensor(labels_train, dtype=torch.float32)
labels_val_tensor = torch.tensor(labels_test, dtype=torch.float32)

# DataLoader准备
train_dataset = TensorDataset(s_minus_train_tensor, bf_vectors_train_tensor, labels_train_tensor)
val_dataset = TensorDataset(s_minus_val_tensor, bf_vectors_val_tensor, labels_val_tensor)

train_loader = DataLoader(dataset=train_dataset, batch_size=1024, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=1024)   # batch_size表示每次迭代传递给模型的样本数目


# 模型定义
class DeepMPDRModel(nn.Module):
    def __init__(self):
        super(DeepMPDRModel, self).__init__()
        self.phi1 = nn.Parameter(torch.randn(181, 64))
        self.phi2 = nn.Parameter(torch.randn(64, 181))
        self.phi3 = nn.Parameter(torch.randn(181, 64))
        self.phi4 = nn.Parameter(torch.randn(181, 181))
        self.phi5 = nn.Parameter(torch.randn(181, 181))
        self.alpha = nn.Parameter(torch.randn(1))
        self.beta = nn.Parameter(torch.randn(1))

    def forward(self, s_minus, bf_vector):
        # 假设 s_minus 和 bf_vector 的前半部分是实部，后半部分是虚部
        split_idx = s_minus.shape[1] // 2
        half_size = bf_vector.shape[1] // 2
        input_s_real = s_minus[:, :split_idx]
        input_s_imag = s_minus[:, split_idx:]
        input_y_real = bf_vector[:, :half_size]
        input_y_imag = bf_vector[:, half_size:]

        # 分别处理实部和虚部
        output_real = self.process(input_s_real, input_y_real)
        output_imag = self.process(input_s_imag, input_y_imag)

        # 计算复数的模
        complex_tensor = torch.complex(output_real, output_imag)
        output_s = torch.abs(complex_tensor)

        return output_s

    def process(self, s_part, bf_vector_part):
        # 补全处理函数，实际逻辑
        output = torch.zeros_like(s_part)
        for i in range(s_part.shape[0]):
            diag_s = torch.diag_embed(s_part[i])  # 对每一行执行 torch.diag_embed
            bf_vector_part_i = bf_vector_part[i].squeeze(-1)  # 移除多余的维度

            t_s = self.phi1 @ self.phi2 @ diag_s @ self.phi3 @ bf_vector_part_i
            b_s = torch.diag(self.phi4 @ diag_s @ self.phi5)  # 计算B_s

            s_tilde = t_s * b_s

            # 将计算结果保存到输出张量中
            output[i] = s_tilde

        # 归一化和软阈值处理
        s_norm = (output - torch.min(output)) / (torch.max(output) - torch.min(output))
        output_s_final = s_norm * torch.sigmoid(self.alpha * (s_norm - self.beta))

        return output_s_final


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepMPDRModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.00001)
loss_func = nn.MSELoss()

num_epochs = 100
train_losses = []
val_losses = []

# Define the path to save the best model
best_model_path = "best_DeepMPDRModel2.pth"
best_val_loss = float('inf')


# 训练模型
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1} Training")
    for s_minus, bf_vector, labels in train_loader_tqdm:
        s_minus, bf_vector, labels = s_minus.to(device), bf_vector.to(device), labels.to(device)
        optimizer.zero_grad()
        s_final = model(s_minus, bf_vector)
        loss = loss_func(s_final, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_loader_tqdm.set_postfix({"Train Loss": loss.item()})
    train_losses.append(train_loss / len(train_loader))

    # 验证阶段
    model.eval()
    val_loss = 0.0
    val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch + 1} Validation")
    with torch.no_grad():
        for s_minus, bf_vector, labels in val_loader_tqdm:
            s_minus, bf_vector, labels = s_minus.to(device), bf_vector.to(device), labels.to(device)
            s_final = model(s_minus, bf_vector)
            loss = loss_func(s_final, labels)
            val_loss += loss.item()
            val_loader_tqdm.set_postfix({"Val Loss": loss.item()})
    val_losses.append(val_loss / len(val_loader))
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"Epoch {epoch + 1}: New best model saved with validation loss {val_loss}")

    print(f'Epoch {epoch + 1}, Train Loss: {train_losses[-1]}, Val Loss: {val_losses[-1]}')

plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
