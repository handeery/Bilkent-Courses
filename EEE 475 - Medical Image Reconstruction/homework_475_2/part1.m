% Part 1.1
figure
plot(ktraj_spiral)
xlabel('k_x')
ylabel('k_y')
title('Interleaved Spiral k-space Trajectory')
axis equal
grid on
%% Part 1.2

area = voronoidens(ktraj_spiral);

% Count NaN values
num_nan = sum(isnan(area(:)));

figure

subplot(2,1,1)
plot(area(:))
xlabel('Sample Index')
ylabel('Voronoi Area')
title('Density Compensation Values (Full Range)')
grid on

subplot(2,1,2)
plot(area(:))
ylim([0 0.005])   
xlabel('Sample Index')
ylabel('Voronoi Area')
title('Density Compensation Values (Zoomed)')
grid on
%% Part 1.3
area_corr = area;

% 1) Replace NaN values with 0
area_corr(isnan(area_corr)) = 0;

% 2) Threshold extremely large values
threshold = prctile(area_corr(:), 99);   % 99th percentile
area_corr(area_corr > threshold) = threshold;

% Plot corrected density in a single figure
figure

subplot(2,1,1)
plot(area_corr(:))
xlabel('Sample Index')
ylabel('Corrected Area')
title('Corrected Density Compensation (Full Range)')
grid on

subplot(2,1,2)
plot(area_corr(:))
ylim([0 threshold])
xlabel('Sample Index')
ylabel('Corrected Area')
title('Corrected Density Compensation (Zoomed)')
grid on

% Verify no NaN remains
num_nan_after = sum(isnan(area_corr(:)))

%% Part 1.4
N = 128;
dx = 1;
tau = N/2;

% Flatten k-space data
kdata_vec = kdata_spiral(:);
ktraj_vec = ktraj_spiral(:);
d_vec = area_corr(:);

kx = real(ktraj_vec);
ky = imag(ktraj_vec);

[x, y] = meshgrid(0:N-1, 0:N-1);

x = x - tau;
y = y - tau;

ima_direct = zeros(N, N);

t_start = cputime;

for p = 1:length(kdata_vec)
    ima_direct = ima_direct + ...
        d_vec(p) * kdata_vec(p) .* ...
        exp(1j*2*pi*(kx(p)*x*dx + ky(p)*y*dx));
end

cpu_time = cputime - t_start;


ima_direct = abs(ima_direct);
ima_direct = ima_direct / max(ima_direct(:));


figure

subplot(2,2,1)
imshow(ima_direct,[])
title('Direct Summation Reconstruction')

subplot(2,2,2)
imshow(abs(ima_direct),[])
title('Magnitude Image')

subplot(2,2,3)
plot(abs(ima_direct(end/2,:)))
title('Central Horizontal Cross-Section')

subplot(2,2,4)
plot(abs(ima_direct(:,end/2)))
title('Central Vertical Cross-Section')

cpu_time

%% Part 1.5
ima_nodcf = zeros(N, N);

t_start = cputime;

for p = 1:length(kdata_vec)
    ima_nodcf = ima_nodcf + ...
        kdata_vec(p) .* ...
        exp(1j*2*pi*(kx(p)*x*dx + ky(p)*y*dx));
end

cpu_time_nodcf = cputime - t_start;

% Normalize
ima_nodcf = abs(ima_nodcf);
ima_nodcf = ima_nodcf / max(ima_nodcf(:));

% Error image
error_nodcf = abs(ima_nodcf - ima_direct);

% Metrics
psnr_nodcf = psnr(ima_nodcf, ima_direct);
ssim_nodcf = ssim(ima_nodcf, ima_direct);

figure

subplot(2,2,1)
imshow(ima_nodcf,[])
title('No Density Compensation')

subplot(2,2,2)
imshow(error_nodcf,[0 0.2])
title('Error Image')

subplot(2,2,3)
plot(abs(ima_nodcf(end/2,:)),'r')
hold on
plot(abs(ima_direct(end/2,:)),'b')
legend('No DCF','Reference')
title('Horizontal Cross-Section')

subplot(2,2,4)
plot(abs(ima_nodcf(:,end/2)),'r')
hold on
plot(abs(ima_direct(:,end/2)),'b')
legend('No DCF','Reference')
title('Vertical Cross-Section')

psnr_nodcf
ssim_nodcf
cpu_time_nodcf

%% Part 1.6
% Double FOV

N2 = 256;
dx = 1;
tau2 = N2/2;

[x2, y2] = meshgrid(0:N2-1, 0:N2-1);
x2 = x2 - tau2;
y2 = y2 - tau2;

ima_doubleFOV = zeros(N2, N2);

t_start = cputime;

for p = 1:length(kdata_vec)
    ima_doubleFOV = ima_doubleFOV + ...
        d_vec(p) * kdata_vec(p) .* ...
        exp(1j*2*pi*(kx(p)*x2*dx + ky(p)*y2*dx));
end

cpu_time_doubleFOV = cputime - t_start;

ima_doubleFOV = abs(ima_doubleFOV);
ima_doubleFOV = ima_doubleFOV / max(ima_doubleFOV(:));

figure
imshow(ima_doubleFOV,[])
title('Double FOV (256x256)')

% Crop central 128x128
start_idx = N2/2 - 64 + 1;
end_idx   = N2/2 + 64;

ima_crop = ima_doubleFOV(start_idx:end_idx, start_idx:end_idx);

% Error vs reference
error_doubleFOV = abs(ima_crop - ima_direct);

psnr_doubleFOV = psnr(ima_crop, ima_direct);
ssim_doubleFOV = ssim(ima_crop, ima_direct);

figure

subplot(2,2,1)
imshow(ima_crop,[])
title('Cropped 128x128')

subplot(2,2,2)
imshow(error_doubleFOV,[0 0.2])
title('Error Image')

subplot(2,2,3)
plot(abs(ima_crop(end/2,:)),'r')
hold on
plot(abs(ima_direct(end/2,:)),'b')
legend('Double FOV','Reference')
title('Horizontal Cross-Section')

subplot(2,2,4)
plot(abs(ima_crop(:,end/2)),'r')
hold on
plot(abs(ima_direct(:,end/2)),'b')
legend('Double FOV','Reference')
title('Vertical Cross-Section')

psnr_doubleFOV
ssim_doubleFOV
cpu_time_doubleFOV

%% Part 1.7
% Parameters
N_half = 256;
dx_half = 0.5;
tau_half = N_half/2;

% Vectorize k-space data (spiral)
kdata_vec = kdata_spiral(:);
ktraj_vec = ktraj_spiral(:);

kx = real(ktraj_vec);
ky = imag(ktraj_vec);

% Density compensation (corrected area from 1.3)
d_vec = area_corr(:);

% Image grid
[xh, yh] = meshgrid(0:N_half-1, 0:N_half-1);
xh = xh - tau_half;
yh = yh - tau_half;

% Direct summation reconstruction
ima_half = zeros(N_half, N_half);

t_start = cputime;

for p = 1:length(kdata_vec)
    ima_half = ima_half + ...
        d_vec(p) * kdata_vec(p) .* ...
        exp(1j*2*pi*(kx(p)*xh*dx_half + ky(p)*yh*dx_half));
end

cpu_time_half = cputime - t_start;

% Magnitude + normalization
ima_half = abs(ima_half);
ima_half = ima_half / max(ima_half(:));

% Zero-pad reference in Fourier domain

% Reference image from 1.4 (128x128)
ref_fft = fftshift(fft2(ima_direct));

pad_fft = zeros(256,256);

start = 256/2 - 64 + 1;
endd  = 256/2 + 64;

pad_fft(start:endd, start:endd) = ref_fft;

ref_half = ifft2(ifftshift(pad_fft));
ref_half = abs(ref_half);
ref_half = ref_half / max(ref_half(:));


error_half = abs(ima_half - ref_half);

psnr_half = psnr(ima_half, ref_half);
ssim_half = ssim(ima_half, ref_half);


figure

subplot(2,2,1)
imshow(ima_half,[])
title('Half Pixel Size (256x256)')

subplot(2,2,2)
imshow(error_half,[0 0.2])
title('Error Image')

subplot(2,2,3)
plot(abs(ima_half(end/2,:)),'r','LineWidth',1.2)
hold on
plot(abs(ref_half(end/2,:)),'b','LineWidth',1.2)
legend('Half Pixel','Reference')
title('Horizontal Cross-Section')

subplot(2,2,4)
plot(abs(ima_half(:,end/2)),'r','LineWidth',1.2)
hold on
plot(abs(ref_half(:,end/2)),'b','LineWidth',1.2)
legend('Half Pixel','Reference')
title('Vertical Cross-Section')

psnr_half
ssim_half
cpu_time_half

%% Part 1.8

N_double = 64;
dx_double = 2;
tau_double = N_double/2;

% Vectorize k-space data
kdata_vec = kdata_spiral(:);
ktraj_vec = ktraj_spiral(:);

kx = real(ktraj_vec);
ky = imag(ktraj_vec);

d_vec = area_corr(:);

% Image grid
[xd, yd] = meshgrid(0:N_double-1, 0:N_double-1);
xd = xd - tau_double;
yd = yd - tau_double;

ima_double = zeros(N_double, N_double);

t_start = cputime;

for p = 1:length(kdata_vec)
    ima_double = ima_double + ...
        d_vec(p) * kdata_vec(p) .* ...
        exp(1j*2*pi*(kx(p)*xd*dx_double + ky(p)*yd*dx_double));
end

cpu_time_double = cputime - t_start;

ima_double = abs(ima_double);
ima_double = ima_double / max(ima_double(:));


double_fft = fftshift(fft2(ima_double));

pad_fft = zeros(128,128);

start = 128/2 - 32 + 1;
endd  = 128/2 + 32;

pad_fft(start:endd, start:endd) = double_fft;

ima_double_pad = ifft2(ifftshift(pad_fft));
ima_double_pad = abs(ima_double_pad);
ima_double_pad = ima_double_pad / max(ima_double_pad(:));

error_double = abs(ima_double_pad - ima_direct);

psnr_double = psnr(ima_double_pad, ima_direct);
ssim_double = ssim(ima_double_pad, ima_direct);

figure

subplot(2,2,1)
imshow(ima_double,[])
title('Double Pixel Size (64x64)')

subplot(2,2,2)
imshow(error_double,[0 0.2])
title('Error Image')

subplot(2,2,3)
plot(abs(ima_double_pad(end/2,:)),'r','LineWidth',1.2)
hold on
plot(abs(ima_direct(end/2,:)),'b','LineWidth',1.2)
legend('Double Pixel','Reference')
title('Horizontal Cross-Section')

subplot(2,2,4)
plot(abs(ima_double_pad(:,end/2)),'r','LineWidth',1.2)
hold on
plot(abs(ima_direct(:,end/2)),'b','LineWidth',1.2)
legend('Double Pixel','Reference')
title('Vertical Cross-Section')

psnr_double
ssim_double
cpu_time_double

%% Part 1.9

% CPU timing
t_start = cputime;

ima_grid_1x = gridkb(kdata_spiral, ...
                     ktraj_spiral, ...
                     area_corr, ...
                     128, ...
                     1, ...
                     2, ...
                     'image');

cpu_time_grid_1x = cputime - t_start;

% Magnitude + normalization
ima_grid_1x = abs(ima_grid_1x);
ima_grid_1x = ima_grid_1x / max(ima_grid_1x(:));

% Error
error_grid_1x = abs(ima_grid_1x - ima_direct);

% Metrics
psnr_grid_1x = psnr(ima_grid_1x, ima_direct);
ssim_grid_1x = ssim(ima_grid_1x, ima_direct);

figure

subplot(2,2,1)
imshow(ima_grid_1x,[])
title('1X Gridding Reconstruction')

subplot(2,2,2)
imshow(error_grid_1x,[0 0.2])
title('Error Image')

subplot(2,2,3)
plot(abs(ima_grid_1x(end/2,:)),'r','LineWidth',1.2)
hold on
plot(abs(ima_direct(end/2,:)),'b','LineWidth',1.2)
legend('1X Gridding','Reference')
title('Horizontal Cross-Section')

subplot(2,2,4)
plot(abs(ima_grid_1x(:,end/2)),'r','LineWidth',1.2)
hold on
plot(abs(ima_direct(:,end/2)),'b','LineWidth',1.2)
legend('1X Gridding','Reference')
title('Vertical Cross-Section')

% Print results
psnr_grid_1x
ssim_grid_1x
cpu_time_grid_1x

%% Part 1.10

% CPU timing
t_start = cputime;

ima_grid_2x_full = gridkb(kdata_spiral, ...
                          ktraj_spiral, ...
                          area_corr, ...
                          128, ...
                          2, ...
                          4, ...
                          'image');

cpu_time_grid_2x = cputime - t_start;

% Magnitude + normalization
ima_grid_2x_full = abs(ima_grid_2x_full);
ima_grid_2x_full = ima_grid_2x_full / max(ima_grid_2x_full(:));

start = 256/2 - 64 + 1;
endd  = 256/2 + 64;

ima_grid_2x = ima_grid_2x_full(start:endd, start:endd);

% Error
error_grid_2x = abs(ima_grid_2x - ima_direct);

% Metrics
psnr_grid_2x = psnr(ima_grid_2x, ima_direct);
ssim_grid_2x = ssim(ima_grid_2x, ima_direct);

figure

subplot(2,2,1)
imshow(ima_grid_2x_full,[])
title('2X Gridding (Full 256x256)')

subplot(2,2,2)
imshow(error_grid_2x,[0 0.2])
title('Error Image (Cropped 128x128)')

subplot(2,2,3)
plot(abs(ima_grid_2x(end/2,:)),'r','LineWidth',1.2)
hold on
plot(abs(ima_direct(end/2,:)),'b','LineWidth',1.2)
legend('2X Gridding','Reference')
title('Horizontal Cross-Section')

subplot(2,2,4)
plot(abs(ima_grid_2x(:,end/2)),'r','LineWidth',1.2)
hold on
plot(abs(ima_direct(:,end/2)),'b','LineWidth',1.2)
legend('2X Gridding','Reference')
title('Vertical Cross-Section')

% Print results
psnr_grid_2x
ssim_grid_2x
cpu_time_grid_2x

%% Part 1.11
% Vectorize k-space data
kdata_vec = kdata_spiral(:);
ktraj_vec = ktraj_spiral(:);

kx = real(ktraj_vec);
ky = imag(ktraj_vec);

N = 128;
kmax = 0.5;

[kx_grid, ky_grid] = meshgrid(linspace(-kmax,kmax,N), ...
                               linspace(-kmax,kmax,N));


F = scatteredInterpolant(kx, ky, kdata_vec, ...
                         'linear', ...
                         'none');

t_start = cputime;

k_cart = F(kx_grid, ky_grid);

cpu_time_scatter_1x = cputime - t_start;

% Replace NaNs with zero (outside ROS)
k_cart(isnan(k_cart)) = 0;

ima_scatter_1x = ifftshift(ifft2(fftshift(k_cart)));
ima_scatter_1x = abs(ima_scatter_1x);
ima_scatter_1x = ima_scatter_1x / max(ima_scatter_1x(:));


error_scatter_1x = abs(ima_scatter_1x - ima_direct);

psnr_scatter_1x = psnr(ima_scatter_1x, ima_direct);
ssim_scatter_1x = ssim(ima_scatter_1x, ima_direct);

figure

subplot(2,2,1)
imshow(ima_scatter_1x,[])
title('Scattered Interpolation (1X)')

subplot(2,2,2)
imshow(error_scatter_1x,[0 0.2])
title('Error Image')

subplot(2,2,3)
plot(abs(ima_scatter_1x(end/2,:)),'r','LineWidth',1.2)
hold on
plot(abs(ima_direct(end/2,:)),'b','LineWidth',1.2)
legend('Scattered','Reference')
title('Horizontal Cross-Section')

subplot(2,2,4)
plot(abs(ima_scatter_1x(:,end/2)),'r','LineWidth',1.2)
hold on
plot(abs(ima_direct(:,end/2)),'b','LineWidth',1.2)
legend('Scattered','Reference')
title('Vertical Cross-Section')

psnr_scatter_1x
ssim_scatter_1x
cpu_time_scatter_1x

%% Part 1.12

% Vectorize k-space data
kdata_vec = kdata_spiral(:);
ktraj_vec = ktraj_spiral(:);

kx = real(ktraj_vec);
ky = imag(ktraj_vec);

N2 = 256;
kmax = 0.5;

[kx_grid2, ky_grid2] = meshgrid(linspace(-kmax,kmax,N2), ...
                                 linspace(-kmax,kmax,N2));


F2 = scatteredInterpolant(kx, ky, kdata_vec, ...
                          'linear', ...
                          'none');

t_start = cputime;

k_cart2 = F2(kx_grid2, ky_grid2);

cpu_time_scatter_2x = cputime - t_start;

% Replace NaNs with zero
k_cart2(isnan(k_cart2)) = 0;

ima_scatter_2x_full = ifftshift(ifft2(fftshift(k_cart2)));
ima_scatter_2x_full = abs(ima_scatter_2x_full);
ima_scatter_2x_full = ima_scatter_2x_full / max(ima_scatter_2x_full(:));

start = 256/2 - 64 + 1;
endd  = 256/2 + 64;

ima_scatter_2x = ima_scatter_2x_full(start:endd, start:endd);

error_scatter_2x = abs(ima_scatter_2x - ima_direct);

psnr_scatter_2x = psnr(ima_scatter_2x, ima_direct);
ssim_scatter_2x = ssim(ima_scatter_2x, ima_direct);

figure

subplot(2,2,1)
imshow(ima_scatter_2x_full,[])
title('Scattered Interpolation (2X Full 256x256)')

subplot(2,2,2)
imshow(error_scatter_2x,[0 0.2])
title('Error Image (Cropped 128x128)')

subplot(2,2,3)
plot(abs(ima_scatter_2x(end/2,:)),'r','LineWidth',1.2)
hold on
plot(abs(ima_direct(end/2,:)),'b','LineWidth',1.2)
legend('Scattered 2X','Reference')
title('Horizontal Cross-Section')

subplot(2,2,4)
plot(abs(ima_scatter_2x(:,end/2)),'r','LineWidth',1.2)
hold on
plot(abs(ima_direct(:,end/2)),'b','LineWidth',1.2)
legend('Scattered 2X','Reference')
title('Vertical Cross-Section')

psnr_scatter_2x
ssim_scatter_2x
cpu_time_scatter_2x