%% Homework 5 - Part 1: Display the Data

% Load data
load('/Users/handeeryilmaz/Downloads/medikal/homework_475_5/multicoil-random.mat');

% im: 256x256x8, mask: 256x256
[Nx, Ny, Nc] = size(im);
fprintf('Image size: %d x %d x %d\n', Nx, Ny, Nc);
fprintf('Mask size: %d x %d\n', size(mask,1), size(mask,2));

% Compute k-space for all coils
M = zeros(Nx, Ny, Nc);
for c = 1:Nc
    M(:,:,c) = fftshift(fft2(ifftshift(im(:,:,c))));
end

% Display magnitude images for all coils
figure('Units','normalized','Position',[0 0 1 0.5]);
montage(abs(im), 'DisplayRange', [], 'Size', [2 4]);
colormap gray;
title('Part 1 - Coil Images (Magnitude)', 'FontSize', 12);

% Display k-space for all coils (log scale)
M_log = log(abs(M) + 1);
figure('Units','normalized','Position',[0 0 1 0.5]);
montage(M_log, 'DisplayRange', [], 'Size', [2 4]);
colormap gray;
title('Part 1 - K-space Spectrums (Log Magnitude)', 'FontSize', 12);

% SoS combination
sos = sqrt(sum(abs(im).^2, 3));

% 95th percentile normalization
p95     = prctile(sos(:), 95);
sos_ref = sos / p95;

% Define ref_val
ref_val = max(sos_ref(:));
fprintf('p95: %.4f\n', p95);
fprintf('ref_val: %.4f\n', ref_val);

% Display SoS reference image
figure('Units','normalized','Position',[0.1 0.1 0.5 0.7]);
imshow(sos_ref, [0 ref_val]);
colormap gray; colorbar;
title('Part 1 - SoS Reference Image (95th pct normalized)', 'FontSize', 12);

fprintf('\nPart 1 complete. ref_val = %.4f\n', ref_val);



%% Part 2 - Sampling Mask and Random Undersampled Images

% Display sampling mask
figure('Units','normalized','Position',[0.1 0.1 0.4 0.5]);
imshow(mask, []);
colormap gray;
title('Part 2 - Sampling Mask', 'FontSize', 12);

% Compute acceleration factor
n_sampled = sum(mask(:));
n_total   = numel(mask);
R         = n_total / n_sampled;
fprintf('Total k-space points:   %d\n', n_total);
fprintf('Sampled k-space points: %d\n', n_sampled);
fprintf('Acceleration factor R:  %.4f\n', R);

% Function: randundersample
function [imu, Mu] = randundersample(im, mask)
    [Nx, Ny, Nc] = size(im);
    Mu  = zeros(Nx, Ny, Nc);
    imu = zeros(Nx, Ny, Nc);

    for c = 1:Nc
        % Centered 2D FFT
        M_full = fftshift(fft2(ifftshift(im(:,:,c))));

        % Apply mask (element-wise multiplication)
        Mu(:,:,c) = M_full .* mask;

        % Zero-fill reconstruction via centered inverse 2D FFT
        imu(:,:,c) = fftshift(ifft2(ifftshift(Mu(:,:,c))));
    end
end

% Apply randundersample
[imu, Mu] = randundersample(im, mask);

fprintf('imu size: %d x %d x %d\n', size(imu,1), size(imu,2), size(imu,3));
fprintf('Mu  size: %d x %d x %d\n', size(Mu,1),  size(Mu,2),  size(Mu,3));

% Display undersampled k-space (log scale)
Mu_log = log(abs(Mu) + 1);
figure('Units','normalized','Position',[0 0 1 0.5]);
montage(Mu_log, 'DisplayRange', [], 'Size', [2 4]);
colormap gray;
title('Part 2 - Undersampled K-space Spectrums (Log Magnitude)', 'FontSize', 12);

% Display zero-fill coil images
figure('Units','normalized','Position',[0 0 1 0.5]);
montage(abs(imu), 'DisplayRange', [], 'Size', [2 4]);
colormap gray;
title('Part 2 - Zero-fill Reconstructed Coil Images (Magnitude)', 'FontSize', 12);

% SoS combination of undersampled images
sos_u   = sqrt(sum(abs(imu).^2, 3));
p95_u   = prctile(sos_u(:), 95);
sos_u_n = sos_u / p95_u;

% Reference image (normalized)
sos_ref = sqrt(sum(abs(im).^2, 3));
sos_ref = sos_ref / prctile(sos_ref(:), 95);

% Display SoS of undersampled images
figure('Units','normalized','Position',[0 0 0.8 0.5]);
subplot(1,2,1);
imshow(sos_u_n, [0 ref_val]);
colormap gray; colorbar;
title('Part 2 - SoS Zero-fill (undersampled)', 'FontSize', 10);

subplot(1,2,2);
error_u = abs(sos_u_n - sos_ref);
imshow(error_u, [0 0.2*ref_val]);
colormap gray; colorbar;
title('Part 2 - Error Image', 'FontSize', 10);
sgtitle('Part 2 - SoS and Error Images');

% PSNR and SSIM
psnr_u = psnr(sos_u_n/ref_val, sos_ref/ref_val);
ssim_u = ssim(sos_u_n/ref_val, sos_ref/ref_val);

fprintf('\nZero-fill Undersampled SoS:\n');
fprintf('PSNR: %.4f dB\n', psnr_u);
fprintf('SSIM: %.4f\n', ssim_u);



%% Part 3 - l1 Regularization in Wavelet Domain

% Add Wavelet class to path
addpath('/Users/handeeryilmaz/Downloads/medikal/homework_475_5/');

% Test
wv = Wavelet('Daubechies', 4, 4);
fprintf('Wavelet operator created successfully.\n');

% Function: l1wavelet
function imth = l1wavelet(imr, beta)
    wv = Wavelet('Daubechies', 4, 4);

    % Forward wavelet transform
    coeffW = wv * imr;

    % Maximum wavelet coefficient magnitude
    w_max = max(abs(coeffW(:)));

    % Threshold value
    thresh = beta * w_max;

    % Soft thresholding on complex coefficients
    abs_coeffW = abs(coeffW);
    coeffW_th  = coeffW ./ (abs_coeffW + eps) .* max(0, abs_coeffW - thresh);

    % Inverse wavelet transform
    imth = wv' * coeffW_th;
end

% Display wavelet coefficients BEFORE thresholding (log scale)
coeff_all = zeros(256, 256, 8);
for c = 1:8
    coeff_all(:,:,c) = wv * imu(:,:,c);
end

figure('Units','normalized','Position',[0 0 1 0.5]);
montage(log(abs(coeff_all) + 1), 'DisplayRange', [], 'Size', [2 4]);
colormap gray;
title('Part 3 - Wavelet Coefficients Before Thresholding (Log Scale)', 'FontSize', 11);

% Test 3 different beta values
beta_vals   = [1e-3, 1e-2, 5e-2];
beta_labels = {'\beta=10^{-3}', '\beta=10^{-2}', '\beta=5\times10^{-2}'};

psnr_beta = zeros(1, 3);
ssim_beta = zeros(1, 3);
im_beta   = cell(1, 3);

for b = 1:3
    beta = beta_vals(b);

    % Apply l1 wavelet regularization to each coil
    imu_th = zeros(256, 256, 8);
    for c = 1:8
        imu_th(:,:,c) = l1wavelet(imu(:,:,c), beta);
    end

    % SoS combination
    sos_th   = sqrt(sum(abs(imu_th).^2, 3));
    p95_th   = prctile(sos_th(:), 95);
    sos_th_n = sos_th / p95_th;

    % Store
    im_beta{b} = sos_th_n;

    % PSNR and SSIM
    psnr_beta(b) = psnr(sos_th_n/ref_val, sos_ref/ref_val);
    ssim_beta(b) = ssim(sos_th_n/ref_val, sos_ref/ref_val);

    fprintf('beta=%.0e | PSNR=%.4f dB | SSIM=%.4f\n', ...
        beta, psnr_beta(b), ssim_beta(b));
end

% Display results for all 3 beta values
figure('Units','normalized','Position',[0 0 1 0.55]);

for b = 1:3
    error_b = abs(im_beta{b} - sos_ref);

    subplot(2, 3, b);
    imshow(im_beta{b}, [0 ref_val]);
    colormap gray; colorbar;
    title(sprintf('%s\nPSNR=%.2f dB, SSIM=%.4f', ...
        beta_labels{b}, psnr_beta(b), ssim_beta(b)), 'FontSize', 8);

    subplot(2, 3, 3+b);
    imshow(error_b, [0 0.2*ref_val]);
    colormap gray; colorbar;
    title(sprintf('Error - %s', beta_labels{b}), 'FontSize', 8);
end

annotation('textbox',[0.01 0.75 0.05 0.05],'String','Recon',...
    'EdgeColor','none','FontWeight','bold');
annotation('textbox',[0.01 0.25 0.05 0.05],'String','Error',...
    'EdgeColor','none','FontWeight','bold');
sgtitle('Part 3 - l1 Wavelet Regularization for 3 \beta Values');

% Summary
fprintf('\n=== Summary: l1 Wavelet Regularization ===\n');
fprintf('%-12s | %-10s | %-10s\n', 'Beta', 'PSNR (dB)', 'SSIM');
fprintf('%s\n', repmat('-', 1, 38));
for b = 1:3
    fprintf('%-12.0e | %-10.4f | %-10.4f\n', ...
        beta_vals(b), psnr_beta(b), ssim_beta(b));
end

%% Part 4 - SPIRiT Kernel Calibration

function Mc = extractCalib(Mu, calibSize)
    [Nx, Ny, ~] = size(Mu);
    cx      = round(Nx/2);
    cy      = round(Ny/2);
    x_calib = cx - floor(calibSize/2) + 1 : cx + floor(calibSize/2);
    y_calib = cy - floor(calibSize/2) + 1 : cy + floor(calibSize/2);
    Mc      = Mu(x_calib, y_calib, :);
end

function kernel = calibrateSpirit(Mc, lambda)
    [calibx, caliby, Nc] = size(Mc);
    kSize  = 3;
    kHalf  = floor(kSize/2);   % = 1
    kk     = kSize * kSize;    % = 9

   
    n_src     = kk * Nc - 1;
    n_patches = (calibx - kSize + 1) * (caliby - kSize + 1);

    fprintf('  n_src (source points per target): %d\n', n_src);
    fprintf('  n_patches:                        %d\n', n_patches);

   
    center_in_patch = kHalf * kSize + (kHalf + 1);  % = 1*3 + 2 = 5

    A = zeros(n_patches, n_src);   % source patches
    B = zeros(n_patches, Nc);      % targets

    patch_idx = 1;
    for px = 1 : calibx - kSize + 1
        for py = 1 : caliby - kSize + 1

            src_vec = zeros(1, n_src);
            col     = 1;

            for c = 1:Nc
                % Extract 3x3 patch for coil c
                patch_c = Mc(px:px+kSize-1, py:py+kSize-1, c);
                p_vec   = patch_c(:)';  % 1 x 9

                if c == 1
                    % Exclude center pixel of coil 1
                    keep       = true(1, kk);
                    keep(center_in_patch) = false;
                    src_vec(col : col + kk - 2) = p_vec(keep);
                    col = col + kk - 1;  % adds 8 elements
                else
                    % Include all 9 pixels of other coils
                    src_vec(col : col + kk - 1) = p_vec;
                    col = col + kk;      % adds 9 elements
                end
            end

            cx_p   = px + kHalf;
            cy_p   = py + kHalf;
            target = squeeze(Mc(cx_p, cy_p, :))';  % 1 x Nc

            A(patch_idx, :) = src_vec;
            B(patch_idx, :) = target;
            patch_idx = patch_idx + 1;
        end
    end

    fprintf('  Calibration matrix A: %d x %d\n', size(A,1), size(A,2));
    fprintf('  Target matrix B:      %d x %d\n', size(B,1), size(B,2));

    kernel = (A'*A + lambda*eye(n_src)) \ (A'*B);

    fprintf('  Kernel size: %d x %d\n', size(kernel,1), size(kernel,2));
end

calibSize = 32;
Mc        = extractCalib(Mu, calibSize);

fprintf('Calibration region Mc: %d x %d x %d\n', ...
    size(Mc,1), size(Mc,2), size(Mc,3));

lambda_spirit = 0;
fprintf('\nCalibrating SPIRiT kernel (lambda=%.0e)...\n', lambda_spirit);
kernel = calibrateSpirit(Mc, lambda_spirit);

fprintf('\nExpected kernel size: %d x %d\n', 9*8-1, 8);
fprintf('Actual   kernel size: %d x %d\n', size(kernel,1), size(kernel,2));

figure('Units','normalized','Position',[0.15 0.15 0.55 0.6]);
imagesc(abs(kernel));
colormap gray; colorbar;
axis image off;
xlabel('Target Coil (1-8)', 'FontSize', 10);
ylabel('Source Point (1-71)', 'FontSize', 10);
title(sprintf('Part 4 - SPIRiT Kernel Magnitude (\\lambda=%.0e)', ...
    lambda_spirit), 'FontSize', 12);
xticks(1:8);
xticklabels({'C1','C2','C3','C4','C5','C6','C7','C8'});
xtickangle(0);

fprintf('\nPart 4 complete.\n');


%% Part 5 - SPIRiT Reconstruction (one iteration, no data consistency)

function [imr, Mr] = spirit(Mu, kernel)
    [Nx, Ny, Nc] = size(Mu);
    kSize  = 3;
    kHalf  = floor(kSize/2);
    kk     = kSize * kSize;

    center_in_patch = kHalf * kSize + (kHalf + 1);  % = 5

    Mu_pad = padarray(Mu, [kHalf, kHalf, 0], 0, 'both');  % (Nx+2) x (Ny+2) x Nc

    Mr = zeros(Nx, Ny, Nc);

    for x = 1:Nx
        for y = 1:Ny
            patch = Mu_pad(x:x+kSize-1, y:y+kSize-1, :);

            src_vec = zeros(1, kk*Nc - 1);
            col = 1;
            for c = 1:Nc
                p_vec = reshape(patch(:,:,c), 1, kk);
                if c == 1
                    keep = true(1, kk);
                    keep(center_in_patch) = false;
                    src_vec(col:col+kk-2) = p_vec(keep);
                    col = col + kk - 1;
                else
                    src_vec(col:col+kk-1) = p_vec;
                    col = col + kk;
                end
            end

            predicted = src_vec * kernel;

            Mr(x, y, :) = predicted;
        end
    end

    imr = zeros(Nx, Ny, Nc);
    for c = 1:Nc
        imr(:,:,c) = fftshift(ifft2(ifftshift(Mr(:,:,c))));
    end
end

lambda_vals = [0, 1e-4, 1e-3, 1e-2];

fprintf('Finding best lambda for SPIRiT...\n');
for lam = lambda_vals
    kernel_test = calibrateSpirit(Mc, lam);
    [imr_test, ~] = spirit(Mu, kernel_test);
    sos_test   = sqrt(sum(abs(imr_test).^2, 3));
    p95_test   = prctile(sos_test(:), 95);
    sos_test_n = sos_test / p95_test;
    psnr_test  = psnr(sos_test_n/ref_val, sos_ref/ref_val);
    ssim_test  = ssim(sos_test_n/ref_val, sos_ref/ref_val);
    fprintf('  lambda=%.0e | PSNR=%.4f dB | SSIM=%.4f\n', ...
        lam, psnr_test, ssim_test);
end

lambda_best = 1e-3;  % adjust based on output above
fprintf('\nUsing lambda=%.0e for SPIRiT kernel...\n', lambda_best);
kernel_spirit = calibrateSpirit(Mc, lambda_best);

fprintf('Applying SPIRiT (1 iteration, no data consistency)...\n');
[imr5, Mr5] = spirit(Mu, kernel_spirit);

sos5   = sqrt(sum(abs(imr5).^2, 3));
p95_5  = prctile(sos5(:), 95);
sos5_n = sos5 / p95_5;

psnr5 = psnr(sos5_n/ref_val, sos_ref/ref_val);
ssim5 = ssim(sos5_n/ref_val, sos_ref/ref_val);
fprintf('SPIRiT 1 iter (no DC): PSNR=%.4f dB | SSIM=%.4f\n', psnr5, ssim5);

figure('Units','normalized','Position',[0 0 1 0.5]);
montage(abs(imr5), 'DisplayRange', [], 'Size', [2 4]);
colormap gray;
title('Part 5 - SPIRiT Coil Images (1 iter, no DC)', 'FontSize', 11);

Mr5_log = log(abs(Mr5) + 1);
figure('Units','normalized','Position',[0 0 1 0.5]);
montage(Mr5_log, 'DisplayRange', [], 'Size', [2 4]);
colormap gray;
title('Part 5 - SPIRiT K-space (1 iter, no DC)', 'FontSize', 11);

figure('Units','normalized','Position',[0 0 0.8 0.5]);
subplot(1,2,1);
imshow(sos5_n, [0 ref_val]);
colormap gray; colorbar;
title(sprintf('Part 5 - SPIRiT SoS\nPSNR=%.2f dB, SSIM=%.4f', ...
    psnr5, ssim5), 'FontSize', 9);

subplot(1,2,2);
error5 = abs(sos5_n - sos_ref);
imshow(error5, [0 0.2*ref_val]);
colormap gray; colorbar;
title('Part 5 - Error Image', 'FontSize', 9);
sgtitle('Part 5 - SPIRiT Reconstruction (1 iter, no data consistency)');

%% Part 6 - One Full SPIRiT Iteration WITH Data Consistency

fprintf('Applying SPIRiT (1 iteration WITH data consistency)...\n');

[imr6, Mr6] = spirit(Mu, kernel_spirit);

Mr6(mask) = Mu(mask);

imr6 = zeros(256, 256, 8);
for c = 1:8
    imr6(:,:,c) = fftshift(ifft2(ifftshift(Mr6(:,:,c))));
end

sos6   = sqrt(sum(abs(imr6).^2, 3));
p95_6  = prctile(sos6(:), 95);
sos6_n = sos6 / p95_6;

psnr6 = psnr(sos6_n/ref_val, sos_ref/ref_val);
ssim6 = ssim(sos6_n/ref_val, sos_ref/ref_val);
fprintf('SPIRiT 1 iter (WITH DC): PSNR=%.4f dB | SSIM=%.4f\n', psnr6, ssim6);

figure('Units','normalized','Position',[0 0 1 0.5]);
montage(abs(imr6), 'DisplayRange', [], 'Size', [2 4]);
colormap gray;
title('Part 6 - SPIRiT Coil Images (1 iter, WITH DC)', 'FontSize', 11);

Mr6_log = log(abs(Mr6) + 1);
figure('Units','normalized','Position',[0 0 1 0.5]);
montage(Mr6_log, 'DisplayRange', [], 'Size', [2 4]);
colormap gray;
title('Part 6 - SPIRiT K-space (1 iter, WITH DC)', 'FontSize', 11);

figure('Units','normalized','Position',[0 0 0.8 0.5]);
subplot(1,2,1);
imshow(sos6_n, [0 ref_val]);
colormap gray; colorbar;
title(sprintf('Part 6 - SPIRiT SoS (WITH DC)\nPSNR=%.2f dB, SSIM=%.4f', ...
    psnr6, ssim6), 'FontSize', 9);

subplot(1,2,2);
error6 = abs(sos6_n - sos_ref);
imshow(error6, [0 0.2*ref_val]);
colormap gray; colorbar;
title('Part 6 - Error Image', 'FontSize', 9);
sgtitle('Part 6 - SPIRiT Reconstruction (1 iter, WITH data consistency)');

fprintf('\n=== Part 5 vs Part 6 Comparison ===\n');
fprintf('%-30s | %-10s | %-10s\n', 'Method', 'PSNR (dB)', 'SSIM');
fprintf('%s\n', repmat('-', 1, 55));
fprintf('%-30s | %-10.4f | %-10.4f\n', 'Zero-fill (Part 2)',    28.9548, 0.8276);
fprintf('%-30s | %-10.4f | %-10.4f\n', 'SPIRiT no DC (Part 5)', psnr5,   ssim5);
fprintf('%-30s | %-10.4f | %-10.4f\n', 'SPIRiT with DC (Part 6)',psnr6,   ssim6);

%% Part 7 - 10 Full SPIRiT Iterations (kernel + data consistency)

fprintf('Running 10 SPIRiT iterations...\n');

Niter = 10;

psnr7 = zeros(1, Niter);
ssim7 = zeros(1, Niter);

% Initialize with undersampled k-space
Mr_iter = Mu;

for iter = 1:Niter
    [imr_iter, Mr_iter] = spirit(Mr_iter, kernel_spirit);
    Mr_iter(mask) = Mu(mask);
    imr_iter = zeros(256, 256, 8);
    for c = 1:8
        imr_iter(:,:,c) = fftshift(ifft2(ifftshift(Mr_iter(:,:,c))));
    end

    sos_iter   = sqrt(sum(abs(imr_iter).^2, 3));
    p95_iter   = prctile(sos_iter(:), 95);
    sos_iter_n = sos_iter / p95_iter;

    psnr7(iter) = psnr(sos_iter_n/ref_val, sos_ref/ref_val);
    ssim7(iter) = ssim(sos_iter_n/ref_val, sos_ref/ref_val);

    fprintf('  Iter %2d | PSNR=%.4f dB | SSIM=%.4f\n', ...
        iter, psnr7(iter), ssim7(iter));
end

imr7_final  = imr_iter;
Mr7_final   = Mr_iter;
sos7_final  = sqrt(sum(abs(imr7_final).^2, 3));
sos7_n      = sos7_final / prctile(sos7_final(:), 95);

fprintf('\nFinal iteration (iter 10):\n');
fprintf('PSNR=%.4f dB | SSIM=%.4f\n', psnr7(end), ssim7(end));

figure('Units','normalized','Position',[0.1 0.1 0.7 0.45]);

subplot(1,2,1);
plot(1:Niter, psnr7, 'b-o', 'LineWidth', 2, 'MarkerSize', 6);
hold on;
yline(28.9548, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Zero-fill');
hold off;
xlabel('Iteration Number', 'FontSize', 10);
ylabel('PSNR (dB)', 'FontSize', 10);
title('Part 7 - PSNR vs Iteration (SPIRiT)', 'FontSize', 10);
legend('SPIRiT', 'Zero-fill baseline', 'Location', 'southeast');
grid on;
xticks(1:Niter);

subplot(1,2,2);
plot(1:Niter, ssim7, 'r-o', 'LineWidth', 2, 'MarkerSize', 6);
hold on;
yline(0.8276, 'b--', 'LineWidth', 1.5, 'DisplayName', 'Zero-fill');
hold off;
xlabel('Iteration Number', 'FontSize', 10);
ylabel('SSIM', 'FontSize', 10);
title('Part 7 - SSIM vs Iteration (SPIRiT)', 'FontSize', 10);
legend('SPIRiT', 'Zero-fill baseline', 'Location', 'southeast');
grid on;
xticks(1:Niter);

sgtitle('Part 7 - SPIRiT: 10 Iterations');

figure('Units','normalized','Position',[0 0 1 0.5]);
montage(abs(imr7_final), 'DisplayRange', [], 'Size', [2 4]);
colormap gray;
title('Part 7 - SPIRiT Coil Images (Iter 10)', 'FontSize', 11);

Mr7_log = log(abs(Mr7_final) + 1);
figure('Units','normalized','Position',[0 0 1 0.5]);
montage(Mr7_log, 'DisplayRange', [], 'Size', [2 4]);
colormap gray;
title('Part 7 - SPIRiT K-space (Iter 10)', 'FontSize', 11);

figure('Units','normalized','Position',[0 0 0.8 0.5]);
subplot(1,2,1);
imshow(sos7_n, [0 ref_val]);
colormap gray; colorbar;
title(sprintf('Part 7 - SPIRiT SoS (Iter 10)\nPSNR=%.2f dB, SSIM=%.4f', ...
    psnr7(end), ssim7(end)), 'FontSize', 9);

subplot(1,2,2);
error7 = abs(sos7_n - sos_ref);
imshow(error7, [0 0.2*ref_val]);
colormap gray; colorbar;
title('Part 7 - Error Image (Iter 10)', 'FontSize', 9);
sgtitle('Part 7 - SPIRiT Final Iteration (Iter 10)');

fprintf('\n=== SPIRiT 10 Iterations Summary ===\n');
fprintf('%-8s | %-10s | %-10s\n', 'Iter', 'PSNR (dB)', 'SSIM');
fprintf('%s\n', repmat('-', 1, 35));
for i = 1:Niter
    fprintf('%-8d | %-10.4f | %-10.4f\n', i, psnr7(i), ssim7(i));
end
fprintf('\nZero-fill baseline: PSNR=28.9548 dB | SSIM=0.8276\n');



%% Part 8 - Iterative l1-SPIRiT Reconstruction

function [imr, Mr] = l1spirit(Mu, mask, kernel, beta, Niter)
    [Nx, Ny, Nc] = size(Mu);

    % Initialize with undersampled k-space
    Mr_cur  = Mu;
    imr_cur = zeros(Nx, Ny, Nc);

    % Output arrays: Nx x Ny x Nc x Niter
    imr = zeros(Nx, Ny, Nc, Niter);
    Mr  = zeros(Nx, Ny, Nc, Niter);

    for iter = 1:Niter
        fprintf('  l1-SPIRiT iter %d/%d\n', iter, Niter);

        Mr_cur(mask) = Mu(mask);

        [imr_cur, Mr_cur] = spirit(Mr_cur, kernel);

        for c = 1:Nc
            imr_cur(:,:,c) = l1wavelet(imr_cur(:,:,c), beta);
        end

        for c = 1:Nc
            Mr_cur(:,:,c) = fftshift(fft2(ifftshift(imr_cur(:,:,c))));
        end

        imr(:,:,:,iter) = imr_cur;
        Mr(:,:,:,iter)  = Mr_cur;
    end
end

beta_vals = [1e-4, 5e-4, 1e-3, 5e-3];
fprintf('Finding best beta for l1-SPIRiT (1 iteration)...\n');

for b = 1:length(beta_vals)
    beta_test = beta_vals(b);

    [imr_test, ~] = l1spirit(Mu, mask, kernel_spirit, beta_test, 1);
    imr_test1 = imr_test(:,:,:,1);

    sos_test   = sqrt(sum(abs(imr_test1).^2, 3));
    p95_test   = prctile(sos_test(:), 95);
    sos_test_n = sos_test / p95_test;

    psnr_test = psnr(sos_test_n/ref_val, sos_ref/ref_val);
    ssim_test = ssim(sos_test_n/ref_val, sos_ref/ref_val);
    fprintf('  beta=%.0e | PSNR=%.4f dB | SSIM=%.4f\n', ...
        beta_test, psnr_test, ssim_test);
end

beta_best = 1e-4;  % adjust based on above output
fprintf('\nUsing beta=%.0e for l1-SPIRiT...\n', beta_best);

fprintf('Running l1-SPIRiT (1 iteration)...\n');
[imr8_all, Mr8_all] = l1spirit(Mu, mask, kernel_spirit, beta_best, 1);

imr8   = imr8_all(:,:,:,1);
Mr8    = Mr8_all(:,:,:,1);

sos8   = sqrt(sum(abs(imr8).^2, 3));
p95_8  = prctile(sos8(:), 95);
sos8_n = sos8 / p95_8;

psnr8 = psnr(sos8_n/ref_val, sos_ref/ref_val);
ssim8 = ssim(sos8_n/ref_val, sos_ref/ref_val);
fprintf('l1-SPIRiT 1 iter: PSNR=%.4f dB | SSIM=%.4f\n', psnr8, ssim8);

figure('Units','normalized','Position',[0 0 1 0.5]);
montage(abs(imr8), 'DisplayRange', [], 'Size', [2 4]);
colormap gray;
title(sprintf('Part 8 - l1-SPIRiT Coil Images (1 iter, \\beta=%.0e)', ...
    beta_best), 'FontSize', 11);

Mr8_log = log(abs(Mr8) + 1);
figure('Units','normalized','Position',[0 0 1 0.5]);
montage(Mr8_log, 'DisplayRange', [], 'Size', [2 4]);
colormap gray;
title(sprintf('Part 8 - l1-SPIRiT K-space (1 iter, \\beta=%.0e)', ...
    beta_best), 'FontSize', 11);

figure('Units','normalized','Position',[0 0 0.8 0.5]);
subplot(1,2,1);
imshow(sos8_n, [0 ref_val]);
colormap gray; colorbar;
title(sprintf('Part 8 - l1-SPIRiT SoS (1 iter)\nPSNR=%.2f dB, SSIM=%.4f', ...
    psnr8, ssim8), 'FontSize', 9);

subplot(1,2,2);
error8 = abs(sos8_n - sos_ref);
imshow(error8, [0 0.2*ref_val]);
colormap gray; colorbar;
title(sprintf('Part 8 - Error Image (\\beta=%.0e)', beta_best), 'FontSize', 9);
sgtitle('Part 8 - l1-SPIRiT Reconstruction (1 iteration)');

fprintf('\n=== Part 8 Summary (1 iteration) ===\n');
fprintf('%-30s | %-10s | %-10s\n', 'Method', 'PSNR (dB)', 'SSIM');
fprintf('%s\n', repmat('-', 1, 55));
fprintf('%-30s | %-10.4f | %-10.4f\n', 'Zero-fill',         28.9548, 0.8276);
fprintf('%-30s | %-10.4f | %-10.4f\n', 'SPIRiT 10 iter',    28.9443, 0.8258);
fprintf('%-30s | %-10.4f | %-10.4f\n', 'l1-SPIRiT 1 iter',  psnr8,   ssim8);


%% Part 9 - 10 Iterations of l1-SPIRiT

fprintf('Running 10 iterations of l1-SPIRiT (beta=%.0e)...\n', beta_best);

Niter = 10;
[imr9_all, Mr9_all] = l1spirit(Mu, mask, kernel_spirit, beta_best, Niter);

psnr9 = zeros(1, Niter);
ssim9 = zeros(1, Niter);

for iter = 1:Niter
    imr9_iter   = imr9_all(:,:,:,iter);
    sos9_iter   = sqrt(sum(abs(imr9_iter).^2, 3));
    p95_9       = prctile(sos9_iter(:), 95);
    sos9_iter_n = sos9_iter / p95_9;

    psnr9(iter) = psnr(sos9_iter_n/ref_val, sos_ref/ref_val);
    ssim9(iter) = ssim(sos9_iter_n/ref_val, sos_ref/ref_val);

    fprintf('  Iter %2d | PSNR=%.4f dB | SSIM=%.4f\n', ...
        iter, psnr9(iter), ssim9(iter));
end

imr9_final = imr9_all(:,:,:,end);
Mr9_final  = Mr9_all(:,:,:,end);
sos9_final = sqrt(sum(abs(imr9_final).^2, 3));
sos9_n     = sos9_final / prctile(sos9_final(:), 95);

fprintf('\nFinal iteration (iter 10):\n');
fprintf('PSNR=%.4f dB | SSIM=%.4f\n', psnr9(end), ssim9(end));

figure('Units','normalized','Position',[0.1 0.1 0.7 0.45]);

subplot(1,2,1);
plot(1:Niter, psnr9,  'b-o', 'LineWidth', 2, 'MarkerSize', 6, ...
    'DisplayName', 'l1-SPIRiT'); hold on;
plot(1:Niter, psnr7,  'g-s', 'LineWidth', 2, 'MarkerSize', 6, ...
    'DisplayName', 'SPIRiT');
yline(28.9548, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Zero-fill');
hold off;
xlabel('Iteration Number', 'FontSize', 10);
ylabel('PSNR (dB)', 'FontSize', 10);
title('Part 9 - PSNR vs Iteration', 'FontSize', 10);
legend('Location', 'southeast', 'FontSize', 8);
grid on; xticks(1:Niter);

subplot(1,2,2);
plot(1:Niter, ssim9, 'b-o', 'LineWidth', 2, 'MarkerSize', 6, ...
    'DisplayName', 'l1-SPIRiT'); hold on;
plot(1:Niter, ssim7, 'g-s', 'LineWidth', 2, 'MarkerSize', 6, ...
    'DisplayName', 'SPIRiT');
yline(0.8276, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Zero-fill');
hold off;
xlabel('Iteration Number', 'FontSize', 10);
ylabel('SSIM', 'FontSize', 10);
title('Part 9 - SSIM vs Iteration', 'FontSize', 10);
legend('Location', 'southeast', 'FontSize', 8);
grid on; xticks(1:Niter);

sgtitle(sprintf('Part 9 - l1-SPIRiT: 10 Iterations (\\beta=%.0e)', beta_best));

figure('Units','normalized','Position',[0 0 0.8 0.5]);
subplot(1,2,1);
imshow(sos9_n, [0 ref_val]);
colormap gray; colorbar;
title(sprintf('Part 9 - l1-SPIRiT SoS (Iter 10)\nPSNR=%.2f dB, SSIM=%.4f', ...
    psnr9(end), ssim9(end)), 'FontSize', 9);

subplot(1,2,2);
error9 = abs(sos9_n - sos_ref);
imshow(error9, [0 0.2*ref_val]);
colormap gray; colorbar;
title(sprintf('Part 9 - Error Image (\\beta=%.0e)', beta_best), 'FontSize', 9);
sgtitle('Part 9 - l1-SPIRiT Final Iteration (Iter 10)');

fprintf('\n========================================\n');
fprintf('     FINAL COMPREHENSIVE SUMMARY\n');
fprintf('========================================\n');
fprintf('%-28s | %-10s | %-10s\n', 'Method', 'PSNR (dB)', 'SSIM');
fprintf('%s\n', repmat('-', 1, 55));
fprintf('%-28s | %-10.4f | %-10.4f\n', 'Zero-fill (Part 2)',       28.9548, 0.8276);
fprintf('%-28s | %-10.4f | %-10.4f\n', 'l1-wavelet b=1e-3 (Pt3)',  28.6854, 0.8258);
fprintf('%-28s | %-10.4f | %-10.4f\n', 'SPIRiT 1iter no DC (Pt5)', 28.4702, 0.8223);
fprintf('%-28s | %-10.4f | %-10.4f\n', 'SPIRiT 1iter+DC (Pt6)',    28.9416, 0.8258);
fprintf('%-28s | %-10.4f | %-10.4f\n', 'SPIRiT 10iter (Pt7)',      28.9443, 0.8258);
fprintf('%-28s | %-10.4f | %-10.4f\n', 'l1-SPIRiT 1iter (Pt8)',    28.4524, 0.8228);
fprintf('%-28s | %-10.4f | %-10.4f\n', 'l1-SPIRiT 10iter (Pt9)',   psnr9(end), ssim9(end));
fprintf('========================================\n');