% 参数设置
N = 64;
lambda = 1; 
d = lambda / 2;  
FOV = -90:1:90;   
SNR_dB = 15;     
num_iterations = 3000;  

% 生成角度网格
theta = FOV;
theta_rad = deg2rad(theta);

% 初始化波束向量矩阵和标签矩阵
M = length(theta);
beamforming_vectors = zeros(num_iterations, N);
labels = zeros(num_iterations, M); % 每个波束向量对应整个角度范围的标签

for iter = 1:num_iterations
    for m = 1:M
        start_angle = randi([FOV(1), FOV(end) - 2]); % 确保三个目标在视场范围内
        DOAs = [start_angle, start_angle + 1, start_angle + 2]; 
        
        g = zeros(1, N);
        label = zeros(1, M); 
        for k = 1:3
            sk = randn(1, 1) + 1i * randn(1, 1); 
            DOA = deg2rad(DOAs(k)); 
            gk = abs(sk) * exp(1i * 2 * pi * d / lambda * sin(DOA) * (1:N)); 
            g = g + gk;
            % 直接计算匹配的DOA索引
            DOA_index = round((rad2deg(DOA) - FOV(1)) + 1);
            label(DOA_index) = abs(sk); % 为匹配的DOA赋值反射系数的幅度
        end
        
        % 保持SNR
        noise_power = 10^(-SNR_dB/10);
        noise = sqrt(noise_power/2) * (randn(1, N) + 1i * randn(1, N));
        g = g + noise;
        % 归一化波束向量
        g = g / norm(g);
        beamforming_vectors((iter-1)*M + m, :) = g;
        labels((iter-1)*M + m, :) = label; % 保存当前波束向量的标签
    end
end

save('beamforming_dataset_extended.mat', 'beamforming_vectors', 'labels');


A = zeros(N, length(theta_rad));
for idx = 1:length(theta_rad)
    theta = theta_rad(idx);
    A(:, idx) = exp(1i * 2 * pi * d / lambda * sin(theta) * (0:N-1)');
end

AH = conj(A');

s_minus_results = zeros(size(beamforming_vectors, 1), length(theta_rad));
for i = 1:size(beamforming_vectors, 1)
    y = beamforming_vectors(i, :).';
    s_minus_results(i, :) = AH * y;
end

save('s_minus_results.mat', 's_minus_results');
