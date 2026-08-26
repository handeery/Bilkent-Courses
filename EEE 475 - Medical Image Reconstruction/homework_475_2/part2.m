
% Part 2.1

P = phantom('Modified Shepp-Logan', 256);

P = P / max(P(:));   % normalize

figure
subplot(2,2,1)
imshow(P,[])
title('Modified Shepp-Logan Phantom (256x256)')

subplot(2,2,2)
plot(P(128,:),'LineWidth',1.5)
title('Central Horizontal Cross-Section')
xlabel('Pixel Index')
ylabel('Intensity')

subplot(2,2,3)
plot(P(:,128),'LineWidth',1.5)
title('Central Vertical Cross-Section')
xlabel('Pixel Index')
ylabel('Intensity')

theta = 0:179;

proj = radon(P, theta);

subplot(2,2,4)
imshow(proj,[])
title('Sinogram')
xlabel('\theta (degrees)')
ylabel('l (detector position)')

size(proj)

%% Part 2.2

t_start = cputime;

ima_fbp_full = iradon(proj, theta);  

cpu_time_fbp = cputime - t_start;

ima_fbp = ima_fbp_full(2:257, 2:257);

% Normalize
ima_fbp = ima_fbp / max(ima_fbp(:));

% Error
error_fbp = abs(ima_fbp - P);

% Metrics
psnr_fbp = psnr(ima_fbp, P);
ssim_fbp = ssim(ima_fbp, P);

figure

subplot(2,2,1)
imshow(ima_fbp,[])
title('Filtered Backprojection (Ram-Lak)')

subplot(2,2,2)
imshow(error_fbp,[0 0.2])
title('Error Image')

subplot(2,2,3)
plot(ima_fbp(128,:),'r','LineWidth',1.2)
hold on
plot(P(128,:),'b','LineWidth',1.2)
legend('FBP','Reference')
title('Horizontal Cross-Section')

subplot(2,2,4)
plot(ima_fbp(:,128),'r','LineWidth',1.2)
hold on
plot(P(:,128),'b','LineWidth',1.2)
legend('FBP','Reference')
title('Vertical Cross-Section')

psnr_fbp
ssim_fbp
cpu_time_fbp

%% Part 2.3
t_start = cputime;

% Reconstruction without filtering
ima_bp_full = iradon(proj, theta, 'linear', 'none');

cpu_time_bp = cputime - t_start;

% Crop central 256x256
ima_bp = ima_bp_full(2:257, 2:257);

% Normalize
ima_bp = ima_bp / max(ima_bp(:));

% Error
error_bp = abs(ima_bp - P);

% Metrics
psnr_bp = psnr(ima_bp, P);
ssim_bp = ssim(ima_bp, P);

figure

subplot(2,2,1)
imshow(ima_bp,[])
title('Naïve Backprojection (No Filter)')

subplot(2,2,2)
imshow(error_bp,[0 0.2])
title('Error Image')

subplot(2,2,3)
plot(ima_bp(128,:),'r','LineWidth',1.2)
hold on
plot(P(128,:),'b','LineWidth',1.2)
legend('BP','Reference')
title('Horizontal Cross-Section')

subplot(2,2,4)
plot(ima_bp(:,128),'r','LineWidth',1.2)
hold on
plot(P(:,128),'b','LineWidth',1.2)
legend('BP','Reference')
title('Vertical Cross-Section')

psnr_bp
ssim_bp
cpu_time_bp


%% Part 2.4

% 1D FFT along detector axis
kdata_proj = fftshift(fft(ifftshift(proj,1),[],1),1);

Nr = size(kdata_proj,1);      % 367
Ntheta = size(kdata_proj,2);  % 180

kr = (-Nr/2:Nr/2-1).' / Nr;   % CORRECT discrete frequency grid

theta_rad = deg2rad(theta);

ktraj_proj = zeros(Nr, Ntheta);

for n = 1:Ntheta
    ktraj_proj(:,n) = kr * cos(theta_rad(n)) + ...
                      1j * kr * sin(theta_rad(n));
end

figure
plot(real(ktraj_proj), imag(ktraj_proj), '.')
xlabel('k_x')
ylabel('k_y')
title('Radial k-space Trajectory (Correct Scaling)')

figure
imagesc(theta, kr, log(abs(kdata_proj)+1))
colormap gray
colorbar
xlabel('\theta (degrees)')
ylabel('k_r')
title('Log Magnitude of k-space Data')

size(kdata_proj)
size(ktraj_proj)

%% Part 2.5
Nr = size(kdata_proj,1);      % 367
Ntheta = size(kdata_proj,2);  % 180

kmax = 0.5;
kr = linspace(-kmax, kmax, Nr).';  % column vector

% Geometric density compensation
dcf_radial = abs(kr);

% Replicate for all angles
dcf_radial = repmat(dcf_radial, 1, Ntheta);

size(dcf_radial)

figure
plot(kr, dcf_radial(:,1),'LineWidth',1.5)
xlabel('k_r')
ylabel('Density Compensation Factor')
title('Radial Density Compensation (First Angle)')

size(dcf_radial)

%% Part 2.6

t_start = cputime;

% FFT normalization
kdata_proj_norm = kdata_proj / Nr;

% Density compensation
kdata_weighted = kdata_proj_norm .* abs(kr);

N = 256;
x = linspace(-0.5,0.5,N);
y = linspace(-0.5,0.5,N);
[X,Y] = meshgrid(x,y);

ima_ds = zeros(N,N);

dkr = abs(kr(2)-kr(1));
dtheta = abs(theta_rad(2)-theta_rad(1));

for n = 1:Ntheta
    theta_val = theta_rad(n);
    for m = 1:Nr
        kr_val = kr(m);
        kx = kr_val * cos(theta_val);
        ky = kr_val * sin(theta_val);

        ima_ds = ima_ds + ...
            kdata_weighted(m,n) * ...
            exp(1j*2*pi*(kx*X + ky*Y)) * ...
            dkr * dtheta;
    end
end

ima_ds = abs(ima_ds);
ima_ds = ima_ds / max(ima_ds(:));

cpu_time_ds = cputime - t_start;

error_ds = abs(ima_ds - P);

psnr_ds = psnr(ima_ds, P);
ssim_ds = ssim(ima_ds, P);

figure

subplot(2,2,1)
imshow(ima_ds,[])
title('Direct Summation (Radial)')

subplot(2,2,2)
imshow(error_ds,[0 0.2])
title('Error Image')

subplot(2,2,3)
plot(ima_ds(128,:),'r','LineWidth',1.2)
hold on
plot(P(128,:),'b','LineWidth',1.2)
legend('Direct','Reference')
title('Horizontal Cross-Section')

subplot(2,2,4)
plot(ima_ds(:,128),'r','LineWidth',1.2)
hold on
plot(P(:,128),'b','LineWidth',1.2)
legend('Direct','Reference')
title('Vertical Cross-Section')

psnr_ds
ssim_ds
cpu_time_ds

%% Part 2.7

t_start = cputime;

% Normalize FFT output
kdata_proj_norm = kdata_proj / Nr;

% Proper DCF size
dcf_radial = repmat(abs(kr), 1, Ntheta);

% 2X gridding
ima_grid_full = gridkb(kdata_proj_norm, ...
                       ktraj_proj, ...
                       dcf_radial, ...
                       256, ...
                       2, ...
                       4, ...
                       'image');

cpu_time_grid = cputime - t_start;

% Crop center 256x256
start = 512/2 - 128 + 1;
endd  = 512/2 + 128;

ima_grid = ima_grid_full(start:endd, start:endd);

ima_grid = abs(ima_grid);
ima_grid = ima_grid / max(ima_grid(:));

error_grid = abs(ima_grid - P);

psnr_grid = psnr(ima_grid, P);
ssim_grid = ssim(ima_grid, P);


figure

subplot(2,2,1)
imshow(ima_grid_full,[])
title('2X Gridding (Full 512x512)')

subplot(2,2,2)
imshow(ima_grid,[])
title('Cropped 256x256')

subplot(2,2,3)
imshow(error_grid,[0 0.2])
title('Error Image')

subplot(2,2,4)
plot(ima_grid(128,:),'r','LineWidth',1.2)
hold on
plot(P(128,:),'b','LineWidth',1.2)
legend('Gridding','Reference')
title('Horizontal Cross-Section')

psnr_grid
ssim_grid
cpu_time_grid