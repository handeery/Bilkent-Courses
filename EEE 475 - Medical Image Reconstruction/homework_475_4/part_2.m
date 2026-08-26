%% Part 2.1 - Undersampled Images Function

function [imu, Mu] = undersample(im, Rx, Ry)
    [Nx, Ny, Nc] = size(im);
    
    % Compute centered 2D FFT for each coil
    M = zeros(Nx, Ny, Nc);
    for c = 1:Nc
        M(:,:,c) = fftshift(fft2(ifftshift(im(:,:,c))));
    end
    
    % Create undersampling mask
    row_indices = 1:Rx:Nx;
    col_indices = 1:Ry:Ny;
    
    % Extract sampled k-space lines and compute aliased images
    Mu = zeros(Nx/Rx, Ny/Ry, Nc);
    imu = zeros(Nx/Rx, Ny/Ry, Nc);
    
    for c = 1:Nc
        Mu(:,:,c) = M(row_indices, col_indices, c);
        imu(:,:,c) = fftshift(ifft2(ifftshift(Mu(:,:,c))));
    end
end

test_cases  = {[1,2], [1,4], [2,2]};
case_labels = {'Rx=1, Ry=2', 'Rx=1, Ry=4', 'Rx=2, Ry=2'};

figure('Units', 'normalized', 'Position', [0 0 1 1]);

for t = 1:3
    Rx = test_cases{t}(1);
    Ry = test_cases{t}(2);
    
    [imu, Mu] = undersample(im, Rx, Ry);
    
    fprintf('Case %s: size(imu) = %d x %d x %d\n', ...
        case_labels{t}, size(imu,1), size(imu,2), size(imu,3));
    
    % Log magnitude of k-space
    kspace_log = log(abs(Mu) + 1);
    
    % Left column: aliased images
    subplot(3, 2, (t-1)*2 + 1);
    montage(abs(imu), 'DisplayRange', [], 'Parent', gca);
    colormap gray;
    title(sprintf('Aliased Images - %s', case_labels{t}), 'FontSize', 8);
    
    % Right column: k-space
    subplot(3, 2, (t-1)*2 + 2);
    montage(kspace_log, 'DisplayRange', [], 'Parent', gca);
    colormap gray;
    title(sprintf('K-space - %s', case_labels{t}), 'FontSize', 8);
end

sgtitle('Part 2.1 - Undersampled Images and K-space for All Cases');

%% Part 2.2 - L2-Regularized SENSE Reconstruction Function

function im_sense = l2sense(imu, map, Rx, Ry, lambda)
    [Nxu, Nyu, Nc] = size(imu);
    Nx = Nxu * Rx;
    Ny = Nyu * Ry;

    im_sense = zeros(Nx, Ny);
    warning('off', 'MATLAB:rankDeficientMatrix');

    for x = 1:Nxu
        for y = 1:Nyu
            src_x = x:Nxu:Nx;
            src_y = y:Nyu:Ny;
            n_src = length(src_x) * length(src_y);

            S = zeros(Nc, n_src);
            idx = 1;
            for ix = 1:length(src_x)
                for iy = 1:length(src_y)
                    S(:, idx) = map(src_x(ix), src_y(iy), :);
                    idx = idx + 1;
                end
            end

            m_alias = squeeze(imu(x, y, :));
            c = (S'*S + lambda*eye(n_src)) \ (S' * m_alias);

            idx = 1;
            for ix = 1:length(src_x)
                for iy = 1:length(src_y)
                    im_sense(src_x(ix), src_y(iy)) = c(idx);
                    idx = idx + 1;
                end
            end
        end
    end

    warning('on', 'MATLAB:rankDeficientMatrix');

    % Take magnitude only — NO normalization inside function
    im_sense = abs(im_sense);
    im_sense(isnan(im_sense)) = 0;
    im_sense(isinf(im_sense)) = 0;
end

[imu_test, ~] = undersample(im, 1, 1);

% Get raw output (no internal normalization)
im_sense_raw = l2sense(imu_test, map1, 1, 1, 0);

% Normalize using SAME p95_map1 as reference image
im_sense_test = im_sense_raw / p95_map1;

% Error image
error_sense_test = abs(im_sense_test - ref_img);

% Display
figure;
subplot(1,2,1);
imshow(im_sense_test, [0 ref_val]);
colormap gray; colorbar;
title('SENSE Reconstruction (Rx=1, Ry=1, \lambda=0)');

subplot(1,2,2);
imshow(error_sense_test, [0 0.2*ref_val]);
colormap gray; colorbar;
title('Error Image vs OLC map1');
sgtitle('Part 2.2 - SENSE Test (Rx=Ry=1, \lambda=0)');

% PSNR and SSIM
sense_test_norm = im_sense_test / ref_val;
ref_norm = ref_img / ref_val;
psnr_sense_test = psnr(sense_test_norm, ref_norm);
ssim_sense_test = ssim(sense_test_norm, ref_norm);

fprintf('SENSE Test (Rx=Ry=1, lambda=0):\n');
fprintf('PSNR: %.4f dB\n', psnr_sense_test);
fprintf('SSIM: %.4f\n', ssim_sense_test);
fprintf('Max absolute difference: %.6e\n', max(abs(im_sense_test(:) - ref_img(:))));

%% Part 2.3 - SENSE Reconstruction without regularization (lambda=0)
% For (Rx,Ry) = (1,2), (1,4), (2,2)

test_cases  = {[1,2], [1,4], [2,2]};
case_labels = {'Rx=1, Ry=2', 'Rx=1, Ry=4', 'Rx=2, Ry=2'};

figure('Units', 'normalized', 'Position', [0 0 1 0.6]);

for t = 1:3
    Rx = test_cases{t}(1);
    Ry = test_cases{t}(2);

    % Generate undersampled images
    [imu, ~] = undersample(im, Rx, Ry);

    % SENSE reconstruction with lambda=0
    im_sense_raw = l2sense(imu, map1, Rx, Ry, 0);

    % 95th percentile normalization
    p95_sense = prctile(im_sense_raw(:), 95);
    im_sense  = im_sense_raw / p95_sense;

    % Error image
    error_img = abs(im_sense - ref_img);

    % PSNR and SSIM
    sense_norm = im_sense / ref_val;
    ref_norm   = ref_img  / ref_val;
    psnr_val   = psnr(sense_norm, ref_norm);
    ssim_val   = ssim(sense_norm, ref_norm);

    fprintf('Case %s | PSNR: %.4f dB | SSIM: %.4f\n', ...
        case_labels{t}, psnr_val, ssim_val);

    % Reconstruction
    subplot(2, 3, t);
    imshow(im_sense, [0 ref_val]);
    colormap gray; colorbar;
    title(sprintf('%s, \\lambda=0\nPSNR=%.2f dB, SSIM=%.4f', ...
        case_labels{t}, psnr_val, ssim_val), 'FontSize', 8);
    axis image;

    % Error image
    subplot(2, 3, 3 + t);
    imshow(error_img, [0 0.2*ref_val]);
    colormap gray; colorbar;
    title(sprintf('Error - %s', case_labels{t}), 'FontSize', 8);
    axis image;
end

annotation('textbox',[0.01 0.75 0.05 0.05],'String','Recon',...
    'EdgeColor','none','FontWeight','bold');
annotation('textbox',[0.01 0.25 0.05 0.05],'String','Error',...
    'EdgeColor','none','FontWeight','bold');
sgtitle('Part 2.3 - SENSE Reconstruction without Regularization (\lambda=0)');

%% Part 2.4 - Final display with confirmed best lambdas

test_cases   = {[1,2],    [1,4],    [2,2]};
case_labels  = {'Rx=1, Ry=2', 'Rx=1, Ry=4', 'Rx=2, Ry=2'};
best_lambdas = [178, 100, 3.16e4];  % confirmed from combined sweep
best_psnrs   = [17.42, 18.71, 17.14];
best_ssims   = [0.4595, 0.4998, 0.4465];

figure('Units', 'normalized', 'Position', [0 0 1 0.6]);

for t = 1:3
    Rx = test_cases{t}(1);
    Ry = test_cases{t}(2);

    [imu, ~]     = undersample(im, Rx, Ry);
    im_sense_raw = l2sense(imu, map1, Rx, Ry, best_lambdas(t));
    p95_sense    = prctile(im_sense_raw(:), 95);
    im_sense     = im_sense_raw / p95_sense;
    error_img    = abs(im_sense - ref_img);

    subplot(2, 3, t);
    imshow(im_sense, [0 ref_val]); colormap gray; colorbar;
    title(sprintf('%s, \\lambda=%.2e\nPSNR=%.2f dB, SSIM=%.4f', ...
        case_labels{t}, best_lambdas(t), best_psnrs(t), best_ssims(t)), ...
        'FontSize', 8);

    subplot(2, 3, 3+t);
    imshow(error_img, [0 0.2*ref_val]); colormap gray; colorbar;
    title(sprintf('Error - %s', case_labels{t}), 'FontSize', 8);
end

annotation('textbox',[0.01 0.75 0.05 0.05],'String','Recon',...
    'EdgeColor','none','FontWeight','bold');
annotation('textbox',[0.01 0.25 0.05 0.05],'String','Error',...
    'EdgeColor','none','FontWeight','bold');
sgtitle('Part 2.4 - SENSE with Optimal Regularization');

%% Part 2.5 - SENSE Reconstruction using map2 (inaccurate sensitivity maps)

test_cases   = {[1,2],    [1,4],    [2,2]};
case_labels  = {'Rx=1, Ry=2', 'Rx=1, Ry=4', 'Rx=2, Ry=2'};
best_lambdas = [178, 100, 3.16e4];  % optimal lambdas from Part 2.4

% Store results
psnr_map1 = [17.42, 18.71, 17.14];  % from Part 2.4
ssim_map1 = [0.4595, 0.4998, 0.4465];

psnr_map2 = zeros(1, 3);
ssim_map2 = zeros(1, 3);

figure('Units', 'normalized', 'Position', [0 0 1 0.85]);

for t = 1:3
    Rx  = test_cases{t}(1);
    Ry  = test_cases{t}(2);
    lam = best_lambdas(t);

    % Generate undersampled images
    [imu, ~] = undersample(im, Rx, Ry);

    % SENSE with map1 (accurate)
    im_map1_raw = l2sense(imu, map1, Rx, Ry, lam);
    p95_map1_s  = prctile(im_map1_raw(:), 95);
    im_map1_s   = im_map1_raw / p95_map1_s;

    % SENSE with map2 (inaccurate)
    im_map2_raw = l2sense(imu, map2, Rx, Ry, lam);
    p95_map2_s  = prctile(im_map2_raw(:), 95);
    im_map2_s   = im_map2_raw / p95_map2_s;

    % Error images vs reference OLC map1
    error_map1 = abs(im_map1_s - ref_img);
    error_map2 = abs(im_map2_s - ref_img);

    % PSNR and SSIM
    psnr_map2(t) = psnr(im_map2_s/ref_val, ref_img/ref_val);
    ssim_map2(t) = ssim(im_map2_s/ref_val, ref_img/ref_val);

    fprintf('Case %s:\n', case_labels{t});
    fprintf('  SENSE map1: PSNR=%.4f dB | SSIM=%.4f\n', psnr_map1(t), ssim_map1(t));
    fprintf('  SENSE map2: PSNR=%.4f dB | SSIM=%.4f\n', psnr_map2(t), ssim_map2(t));

    % Row 1: map1 reconstruction
    subplot(3, 4, (t-1)*4 + 1);
    imshow(im_map1_s, [0 ref_val]); colormap gray; colorbar;
    title(sprintf('map1 - %s\nPSNR=%.2f, SSIM=%.4f', ...
        case_labels{t}, psnr_map1(t), ssim_map1(t)), 'FontSize', 7);

    % Row 2: map1 error
    subplot(3, 4, (t-1)*4 + 2);
    imshow(error_map1, [0 0.2*ref_val]); colormap gray; colorbar;
    title(sprintf('Error map1 - %s', case_labels{t}), 'FontSize', 7);

    % Row 3: map2 reconstruction
    subplot(3, 4, (t-1)*4 + 3);
    imshow(im_map2_s, [0 ref_val]); colormap gray; colorbar;
    title(sprintf('map2 - %s\nPSNR=%.2f, SSIM=%.4f', ...
        case_labels{t}, psnr_map2(t), ssim_map2(t)), 'FontSize', 7);

    % Row 4: map2 error
    subplot(3, 4, (t-1)*4 + 4);
    imshow(error_map2, [0 0.2*ref_val]); colormap gray; colorbar;
    title(sprintf('Error map2 - %s', case_labels{t}), 'FontSize', 7);
end

sgtitle('Part 2.5 - SENSE Reconstruction: map1 vs map2');

fprintf('\n=== Summary: map1 vs map2 SENSE ===\n');
fprintf('%-15s | %-12s | %-10s | %-12s | %-10s\n', ...
    'Case', 'PSNR map1', 'SSIM map1', 'PSNR map2', 'SSIM map2');
fprintf('%s\n', repmat('-', 1, 70));
for t = 1:3
    fprintf('%-15s | %-12.4f | %-10.4f | %-12.4f | %-10.4f\n', ...
        case_labels{t}, psnr_map1(t), ssim_map1(t), ...
        psnr_map2(t), ssim_map2(t));
end

%% Part 2.7 - Corrected G-factor using standard formula

function g_map = gfactor(map, Rx, Ry, lambda)
    [Nx, Ny, Nc] = size(map);
    Nxu   = Nx / Rx;
    Nyu   = Ny / Ry;
    n_src = Rx * Ry;

    g_map = zeros(Nx, Ny);

    warning('off', 'MATLAB:singularMatrix');
    warning('off', 'MATLAB:rankDeficientMatrix');
    warning('off', 'MATLAB:nearlySingularMatrix');

    for x = 1:Nxu
        for y = 1:Nyu
            src_x = x:Nxu:Nx;
            src_y = y:Nyu:Ny;

            % Build sensitivity matrix S (Nc x n_src)
            S = zeros(Nc, n_src);
            col = 1;
            for ix = 1:length(src_x)
                for iy = 1:length(src_y)
                    S(:, col) = map(src_x(ix), src_y(iy), :);
                    col = col + 1;
                end
            end

            SHS = S' * S;

            % Skip background voxels
            if norm(SHS, 'fro') < 1e-10
                continue;
            end

            A     = SHS + lambda * eye(n_src);
            A_inv = pinv(A);

            % Standard g-factor formula:
            % g_i = sqrt( [A_inv]_ii * [SHS]_ii ) * sqrt(n_src)
            col = 1;
            for ix = 1:length(src_x)
                for iy = 1:length(src_y)
                    gval = sqrt(n_src) * ...
                           sqrt(real(A_inv(col,col)) * real(SHS(col,col)));
                    if isfinite(gval) && gval >= 0
                        g_map(src_x(ix), src_y(iy)) = gval;
                    end
                    col = col + 1;
                end
            end
        end
    end

    warning('on', 'MATLAB:singularMatrix');
    warning('on', 'MATLAB:rankDeficientMatrix');
    warning('on', 'MATLAB:nearlySingularMatrix');
end


test_cases  = {[1,2],  [1,4],  [2,2]};
case_labels = {'Rx=1, Ry=2', 'Rx=1, Ry=4', 'Rx=2, Ry=2'};
lambda_gfac = 1e-6;

g_maps = cell(1,3);

figure('Units', 'normalized', 'Position', [0 0 1 0.4]);

for t = 1:3
    Rx = test_cases{t}(1);
    Ry = test_cases{t}(2);

    fprintf('Computing g-factor map for %s...\n', case_labels{t});
    g_map      = gfactor(map1, Rx, Ry, lambda_gfac);
    g_maps{t}  = g_map;

    % Brain statistics — exclude background (g=0)
    g_brain = g_map(g_map > 0.01);
    fprintf('  Mean g = %.4f | Max g = %.4f | Median g = %.4f\n', ...
        mean(g_brain), max(g_brain), median(g_brain));

    % Clip extreme values for display
    g_display = g_map;
    clip_val  = prctile(g_brain, 99);  % clip top 1% for display
    g_display(g_display > clip_val) = clip_val;

    subplot(1, 3, t);
    imagesc(g_display);
    colormap hot; colorbar;
    axis image off;
    caxis([0 clip_val]);
    title(sprintf('%s\nMean g=%.2f, Max g=%.2f', ...
        case_labels{t}, mean(g_brain), max(g_brain)), 'FontSize', 9);
end

sgtitle('Part 2.7 - G-factor Maps (map1, \lambda\approx0)');


%% Part 2.8 - G-factor Maps with Optimal Lambda scaled by 1e-6

test_cases   = {[1,2],    [1,4],    [2,2]};
case_labels  = {'Rx=1, Ry=2', 'Rx=1, Ry=4', 'Rx=2, Ry=2'};
best_lambdas = [178, 100, 3.16e4];

% Scale lambdas by 1e-6
lambda_scaled = best_lambdas * 1e-6;

figure('Units', 'normalized', 'Position', [0 0 1 0.8]);

for t = 1:3
    Rx  = test_cases{t}(1);
    Ry  = test_cases{t}(2);
    lam = lambda_scaled(t);

    fprintf('Computing g-factor map for %s (lambda=%.4e)...\n', ...
        case_labels{t}, lam);

    % G-factor with scaled lambda
    g_map_scaled = gfactor(map1, Rx, Ry, lam);

    % G-factor with lambda~0 (from Part 2.7 for comparison)
    g_map_zero   = gfactor(map1, Rx, Ry, 1e-6);

    % Brain statistics
    g_brain_scaled = g_map_scaled(g_map_scaled > 0.01);
    g_brain_zero   = g_map_zero(g_map_zero > 0.01);

    fprintf('  lambda~0   : Mean g=%.4f | Max g=%.4f | Median g=%.4f\n', ...
        mean(g_brain_zero), max(g_brain_zero), median(g_brain_zero));
    fprintf('  lambda*1e-6: Mean g=%.4f | Max g=%.4f | Median g=%.4f\n', ...
        mean(g_brain_scaled), max(g_brain_scaled), median(g_brain_scaled));

    % Clip for display
    clip_zero   = prctile(g_brain_zero,   99);
    clip_scaled = prctile(g_brain_scaled, 99);

    % Row 1: lambda~0 (Part 2.7)
    subplot(2, 3, t);
    g_disp = g_map_zero;
    g_disp(g_disp > clip_zero) = clip_zero;
    imagesc(g_disp, [0 clip_zero]);
    colormap hot; colorbar; axis image off;
    title(sprintf('%s, \\lambda\\approx0\nMean g=%.2f, Max g=%.2f', ...
        case_labels{t}, mean(g_brain_zero), max(g_brain_zero)), 'FontSize', 8);

    % Row 2: lambda*1e-6
    subplot(2, 3, 3 + t);
    g_disp = g_map_scaled;
    g_disp(g_disp > clip_scaled) = clip_scaled;
    imagesc(g_disp, [0 clip_scaled]);
    colormap hot; colorbar; axis image off;
    title(sprintf('%s, \\lambda_{opt}\\times10^{-6}\nMean g=%.2f, Max g=%.2f', ...
        case_labels{t}, mean(g_brain_scaled), max(g_brain_scaled)), 'FontSize', 8);
end

annotation('textbox',[0.01 0.75 0.08 0.05],'String','\lambda\approx0',...
    'EdgeColor','none','FontWeight','bold','FontSize',9);
annotation('textbox',[0.01 0.25 0.08 0.05],'String','\lambda_{opt}\times10^{-6}',...
    'EdgeColor','none','FontWeight','bold','FontSize',9);

sgtitle('Part 2.8 - G-factor Maps: \lambda\approx0 vs \lambda_{opt}\times10^{-6}');


