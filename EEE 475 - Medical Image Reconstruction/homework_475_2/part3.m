% Part 3.1

R = load('shepplogan_radial_data.mat');

kdata_radial = R.kdata;
ktraj_radial = R.ktraj;

size(kdata_radial)
size(ktraj_radial)

figure('Position',[100 100 600 600])
plot(real(ktraj_radial), imag(ktraj_radial), '.')
axis equal
xlabel('k_x')
ylabel('k_y')
title('Radial k-space Trajectory')
grid on


%% Part 3.2

% Radial distance
kr = abs(ktraj_radial);

% Geometric DCF (for visualization only)
dcf_radial = kr;

size(dcf_radial)

figure
plot(dcf_radial(:,1),'LineWidth',2)
xlabel('Sample Index')
ylabel('Density Compensation Value')
title('Density Compensation Along One Radial Line')
grid on


%% Part 3.3

N = 256;

x = (-N/2:N/2-1)/N;
[X,Y] = meshgrid(x,x);

kx = real(ktraj_radial);
ky = imag(ktraj_radial);

% Continuous integral form (include dr and dtheta)
dr = abs(kr(2,1) - kr(1,1));
dtheta = pi / size(kr,2);

dcf_correct = kr * dr * dtheta;

tic

ima_ds = zeros(N,N);

for m = 1:size(kdata_radial,1)
    for n = 1:size(kdata_radial,2)
        
        ima_ds = ima_ds + ...
            dcf_correct(m,n) .* kdata_radial(m,n) .* ...
            exp(1i*2*pi*(kx(m,n)*X + ky(m,n)*Y));
    end
end

cpu_time_ds = toc;

ima_ds = abs(ima_ds);
ima_ds = ima_ds / max(ima_ds(:));

P = phantom(N);
P = P / max(P(:));

ima_ds = flipud(ima_ds);

psnr_ds = psnr(ima_ds,P);
ssim_ds = ssim(ima_ds,P);

figure
subplot(2,2,1)
imshow(ima_ds,[])
title('Direct Summation')

subplot(2,2,2)
imshow(abs(ima_ds-P),[0 0.2])
title('Error Image')

subplot(2,2,3)
plot(ima_ds(N/2,:),'r'); hold on
plot(P(N/2,:),'b')
legend('Direct','Reference')
title('Horizontal Cross-Section')

subplot(2,2,4)
plot(ima_ds(:,N/2),'r'); hold on
plot(P(:,N/2),'b')
legend('Direct','Reference')
title('Vertical Cross-Section')

psnr_ds
ssim_ds
cpu_time_ds


%% Part 3.4

N = 256;
oversamp = 2;
kernel_width = 4;

% --- CORRECT DCF FOR GRIDDING ---
dcf_grid = abs(ktraj_radial);
dcf_grid = dcf_grid / max(dcf_grid(:));   % normalize
% ---------------------------------

tic

ima_grid_full = gridkb(kdata_radial, ...
                       ktraj_radial, ...
                       dcf_grid, ...
                       N, ...
                       oversamp, ...
                       kernel_width, ...
                       'image');

cpu_time_grid = toc;

ima_grid_full = abs(ima_grid_full);

% Crop central 256x256
center = size(ima_grid_full,1)/2;
half = N/2;

ima_grid_crop = ima_grid_full(center-half+1:center+half, ...
                               center-half+1:center+half);

ima_grid_crop = ima_grid_crop / max(ima_grid_crop(:));

P = phantom(N);
P = P / max(P(:));

ima_grid_crop = flipud(ima_grid_crop);

psnr_grid = psnr(ima_grid_crop, P);
ssim_grid = ssim(ima_grid_crop, P);

figure('Position',[100 100 1000 800])

subplot(2,2,1)
imshow(ima_grid_full,[])
title('2X Gridding (Full 512x512)')

subplot(2,2,2)
imshow(ima_grid_crop,[])
title('Cropped 256x256')

subplot(2,2,3)
imshow(abs(ima_grid_crop - P), [0 0.2])
title('Error Image')

subplot(2,2,4)
plot(ima_grid_crop(N/2,:),'r','LineWidth',1.5)
hold on
plot(P(N/2,:),'b','LineWidth',1.5)
legend('Gridding','Reference')
title('Horizontal Cross-Section')

psnr_grid
ssim_grid
cpu_time_grid