import torch
import torch.nn as nn

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
            output[i] = s_tilde  # 根据需要调整

        # 归一化和软阈值处理
        s_norm = (output - torch.min(output)) / (torch.max(output) - torch.min(output))
        output_s_final = s_norm * torch.sigmoid(self.alpha * (s_norm - self.beta))

        return output_s_final
