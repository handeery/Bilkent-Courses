% Load data
load('/Users/handeeryilmaz/Downloads/medikal/homework_475_4/multicoil-data.mat');

fprintf('Size of im: %d x %d x %d\n', size(im,1), size(im,2), size(im,3));
fprintf('Size of map1: %d x %d x %d\n', size(map1,1), size(map1,2), size(map1,3));
fprintf('Size of map2: %d x %d x %d\n', size(map2,1), size(map2,2), size(map2,3));

Nx = size(im,1);   % 224
Ny = size(im,2);   % 160
Nc = size(im,3);   % 8

%% Part 1.1 - Compute k-space
kspace = zeros(Nx, Ny, Nc);
for c = 1:Nc
    kspace(:,:,c) = fftshift(fft2(ifftshift(im(:,:,c))));
end
kspace_display = log(abs(kspace) + 1);

figure('Units', 'normalized', 'Position', [0 0 1 1]);

% --- Top row (4 subplots) ---
subplot(2, 4, 1);
montage(abs(im), 'DisplayRange', [], 'Parent', gca);
colormap gray;
title('Magnitude - Coil Images', 'FontSize', 8);

subplot(2, 4, 2);
montage(angle(im), 'DisplayRange', [-pi pi], 'Parent', gca);
colormap gray;
title('Phase - Coil Images', 'FontSize', 8);

subplot(2, 4, 3);
montage(kspace_display, 'DisplayRange', [], 'Parent', gca);
colormap gray;
title('K-space (log magnitude)', 'FontSize', 8);

subplot(2, 4, 4);
montage(abs(map1), 'DisplayRange', [], 'Parent', gca);
colormap gray;
title('Magnitude - map1', 'FontSize', 8);

% --- Bottom row (3 subplots) ---
subplot(2, 4, 5);
montage(angle(map1), 'DisplayRange', [-pi pi], 'Parent', gca);
colormap gray;
title('Phase - map1', 'FontSize', 8);

subplot(2, 4, 6);
montage(abs(map2), 'DisplayRange', [], 'Parent', gca);
colormap gray;
title('Magnitude - map2', 'FontSize', 8);

subplot(2, 4, 7);
montage(angle(map2), 'DisplayRange', [-pi pi], 'Parent', gca);
colormap gray;
title('Phase - map2', 'FontSize', 8);

% Subplot 8: empty
subplot(2, 4, 8);
axis off;

sgtitle('Part 1.1 - MRI Images, K-space and Coil Sensitivities');

%% Part 1.2 - Optimally Weighted Linear Combination (OLC) using map1

% OLC formula: m_olc = sum(conj(s_c) * m_c) / sum(|s_c|^2)
% computed voxel by voxel across all coils

% Numerator: sum over coils of conj(sensitivity) * image
numerator = sum(conj(map1) .* im, 3);

% Denominator: sum of squares of sensitivities across coils
denominator = sum(abs(map1).^2, 3);

% OLC reconstruction
olc_map1 = numerator ./ denominator;

% Set NaN pixels to zero (where denominator = 0 i.e. background)
olc_map1(isnan(olc_map1)) = 0;

% Take magnitude
olc_map1_mag = abs(olc_map1);

% 95th percentile normalization
p95_map1 = prctile(olc_map1_mag(:), 95);
olc_map1_norm = olc_map1_mag / p95_map1;

% Define ref_val as maximum pixel intensity of normalized OLC map1
% This is the reference image for Parts I and II
ref_img = olc_map1_norm;
ref_val = max(ref_img(:));
fprintf('ref_val (max pixel intensity of OLC map1): %.4f\n', ref_val);
fprintf('p95_map1: %.4f\n', p95_map1);

% Display magnitude image
figure;
imshow(olc_map1_norm, [0 ref_val]);
colormap gray; colorbar;
title('OLC Reconstruction - map1 (magnitude)');

%% Part 1.3 - OLC Reconstruction using map2

% Numerator: sum over coils of conj(map2 sensitivity) * image
numerator_map2 = sum(conj(map2) .* im, 3);

% Denominator: sum of squares of map2 sensitivities
denominator_map2 = sum(abs(map2).^2, 3);

% OLC reconstruction using map2
olc_map2 = numerator_map2 ./ denominator_map2;

% Set NaN pixels to zero
olc_map2(isnan(olc_map2)) = 0;

% Take magnitude
olc_map2_mag = abs(olc_map2);

% 95th percentile normalization
p95_map2 = prctile(olc_map2_mag(:), 95);
olc_map2_norm = olc_map2_mag / p95_map2;

fprintf('p95_map2: %.4f\n', p95_map2);

% Compute error image
error_map2 = abs(olc_map2_norm - ref_img);

% Display magnitude image
figure;
subplot(1, 2, 1);
imshow(olc_map2_norm, [0 ref_val]);
colormap gray; colorbar;
title('OLC Reconstruction - map2 (magnitude)');

% Display error image
subplot(1, 2, 2);
imshow(error_map2, [0 0.2*ref_val]);
colormap gray; colorbar;
title('Error Image - map2 vs map1');

sgtitle('Part 1.3 - OLC using map2');

% Compute PSNR and SSIM
% Normalize both images by ref_val before computing metrics
olc_map2_norm2 = olc_map2_norm / ref_val;
ref_norm = ref_img / ref_val;

psnr_map2 = psnr(olc_map2_norm2, ref_norm);
ssim_map2 = ssim(olc_map2_norm2, ref_norm);

fprintf('PSNR (OLC map2 vs map1): %.4f dB\n', psnr_map2);
fprintf('SSIM (OLC map2 vs map1): %.4f\n', ssim_map2);

%% Part 1.4 - Sum of Squares (SoS) Reconstruction

% SoS formula: m_sos = sqrt(sum(|m_c|^2))
sos = sqrt(sum(abs(im).^2, 3));

% 95th percentile normalization
p95_sos = prctile(sos(:), 95);
sos_norm = sos / p95_sos;

fprintf('p95_sos: %.4f\n', p95_sos);

% Compute error image with respect to reference (OLC map1)
error_sos = abs(sos_norm - ref_img);

% Display magnitude image
figure;
subplot(1, 2, 1);
imshow(sos_norm, [0 ref_val]);
colormap gray; colorbar;
title('SoS Reconstruction (magnitude)');

% Display error image
subplot(1, 2, 2);
imshow(error_sos, [0 0.2*ref_val]);
colormap gray; colorbar;
title('Error Image - SoS vs OLC map1');

sgtitle('Part 1.4 - Sum of Squares Reconstruction');

% Compute PSNR and SSIM
sos_norm2 = sos_norm / ref_val;
ref_norm = ref_img / ref_val;

psnr_sos = psnr(sos_norm2, ref_norm);
ssim_sos = ssim(sos_norm2, ref_norm);

fprintf('PSNR (SoS vs OLC map1): %.4f dB\n', psnr_sos);
fprintf('SSIM (SoS vs OLC map1): %.4f\n', ssim_sos);