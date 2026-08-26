%% Part 3.1 - Generate Undersampled and Calibration Data

function [imc, Mc] = imcalib(im, calibx, caliby)
    % Extract calibration data from center of k-space
    % Returns calibration images imc and k-space Mc
    [Nx, Ny, Nc] = size(im);

    % Compute full k-space
    M = zeros(Nx, Ny, Nc);
    for c = 1:Nc
        M(:,:,c) = fftshift(fft2(ifftshift(im(:,:,c))));
    end

    % Extract central calibration region
    cx = round(Nx/2);
    cy = round(Ny/2);
    x_calib = cx - floor(calibx/2) : cx + ceil(calibx/2) - 1;
    y_calib = cy - floor(caliby/2) : cy + ceil(caliby/2) - 1;

    % Calibration k-space
    Mc = M(x_calib, y_calib, :);

    % Calibration images via inverse FFT
    imc = zeros(calibx, caliby, Nc);
    for c = 1:Nc
        imc(:,:,c) = fftshift(ifft2(ifftshift(Mc(:,:,c))));
    end
end

function [imu, Mu] = undersamplecalib(im, Rx, Ry, calibx, caliby)
    % Undersample k-space but keep calibration region fully sampled
    [Nx, Ny, Nc] = size(im);

    % Compute full k-space
    M = zeros(Nx, Ny, Nc);
    for c = 1:Nc
        M(:,:,c) = fftshift(fft2(ifftshift(im(:,:,c))));
    end

    % Center of k-space
    cx = round(Nx/2);
    cy = round(Ny/2);
    x_calib = cx - floor(calibx/2) : cx + ceil(calibx/2) - 1;
    y_calib = cy - floor(caliby/2) : cy + ceil(caliby/2) - 1;

    % Create undersampling mask
    mask = zeros(Nx, Ny);
    row_indices = 1:Rx:Nx;
    col_indices = 1:Ry:Ny;
    mask(row_indices, col_indices) = 1;

    % Always keep calibration region fully sampled
    mask(x_calib, y_calib) = 1;

    % Apply mask
    Mu = zeros(Nx, Ny, Nc);
    for c = 1:Nc
        Mu(:,:,c) = M(:,:,c) .* mask;
    end

    % Aliased images via inverse FFT of undersampled k-space
    % Extract only sampled rows/cols for output
    Mu_sub = Mu(row_indices, col_indices, :);
    imu = zeros(Nx/Rx, Ny/Ry, Nc);
    for c = 1:Nc
        imu(:,:,c) = fftshift(ifft2(ifftshift(Mu_sub(:,:,c))));
    end
end


% Calibration parameters
calibx = 24;
caliby = 24;

% Get calibration data
[imc, Mc] = imcalib(im, calibx, caliby);

fprintf('Calibration data:\n');
fprintf('  imc size: %d x %d x %d\n', size(imc,1), size(imc,2), size(imc,3));
fprintf('  Mc  size: %d x %d x %d\n', size(Mc,1),  size(Mc,2),  size(Mc,3));

% Test undersampled + calib for (Rx,Ry)=(1,2)
[imu_c, Mu_c] = undersamplecalib(im, 1, 2, calibx, caliby);
fprintf('Undersampled+calib (Rx=1,Ry=2):\n');
fprintf('  imu size: %d x %d x %d\n', size(imu_c,1), size(imu_c,2), size(imu_c,3));
fprintf('  Mu  size: %d x %d x %d\n', size(Mu_c,1),  size(Mu_c,2),  size(Mu_c,3));

% Display calibration images and k-space
figure('Units','normalized','Position',[0 0 1 0.5]);

subplot(1,3,1);
montage(abs(imc), 'DisplayRange', [], 'Parent', gca);
colormap gray;
title(sprintf('Calibration Images (%dx%d)', calibx, caliby), 'FontSize', 9);

subplot(1,3,2);
montage(log(abs(Mc)+1), 'DisplayRange', [], 'Parent', gca);
colormap gray;
title('Calibration K-space (log mag)', 'FontSize', 9);

subplot(1,3,3);
% Show full k-space mask for (Rx=1,Ry=2)
[Nx, Ny, Nc] = size(im);
M_full = zeros(Nx, Ny);
for c = 1:Nc
    M_full = M_full + abs(fftshift(fft2(ifftshift(im(:,:,c)))));
end
mask_show = zeros(Nx, Ny);
mask_show(1:1:Nx, 1:2:Ny) = 1;
mask_show(round(Nx/2)-11:round(Nx/2)+12, ...
          round(Ny/2)-11:round(Ny/2)+12) = 1;
imagesc(mask_show); colormap gray; axis image off;
title('K-space Mask (Rx=1,Ry=2 + calib region)', 'FontSize', 9);

sgtitle('Part 3.1 - Calibration and Undersampled Data');
%% Part 3.2 - GRAPPA Kernel Calibration

function kernel = calibrate(Mc, lambda)

    [calibx, caliby, Nc] = size(Mc);

    % Kernel size (number of source points in each direction)
    kx = 3;  % kernel size in x (rows)
    ky = 4;  % kernel size in y (cols) — covers Ry source points

    n_src_pts = kx * ky * Nc;   % number of source points per kernel
    n_patches = (calibx - kx + 1) * (caliby - ky + 1);

    % Build source matrix A and target matrix B
    A = zeros(n_patches, n_src_pts);
    B = zeros(n_patches, Nc);

    patch_idx = 1;
    for px = 1 : calibx - kx + 1
        for py = 1 : caliby - ky + 1
            % Extract source patch
            patch = Mc(px:px+kx-1, py:py+ky-1, :);
            A(patch_idx, :) = patch(:)';

            % Target: center point of patch for each coil
            cx = px + floor(kx/2);
            cy = py + floor(ky/2);
            B(patch_idx, :) = squeeze(Mc(cx, cy, :))';

            patch_idx = patch_idx + 1;
        end
    end

    % Solve regularized least squares: A * kernel = B
    % kernel = (A'A + lambda*I)^-1 * A'B
    kernel = (A'*A + lambda*eye(n_src_pts)) \ (A'*B);

    fprintf('Kernel size: %d x %d (source pts x target coils)\n', ...
        size(kernel,1), size(kernel,2));
    fprintf('Calibration matrix A: %d x %d\n', size(A,1), size(A,2));
end

calibx   = 24;
caliby   = 24;
lambda_g = 0.01;

[imc, Mc] = imcalib(im, calibx, caliby);

fprintf('Calibrating GRAPPA kernel...\n');
kernel = calibrate(Mc, lambda_g);

fprintf('Kernel calibrated successfully.\n');
fprintf('kernel size: %d x %d\n', size(kernel,1), size(kernel,2));

%% Part 3.3 - GRAPPA Reconstruction

function [imr, Mr] = grappa(Mu, Mc, lambda)

    [Nx, Ny, Nc] = size(Mu);
    [calibx, caliby, ~] = size(Mc);

    % Step 1: Calibrate kernel from Mc
    kernel = calibrate(Mc, lambda);

    kx = 3;
    ky = 4;

    % Step 2: Detect sampled pattern from Mu
    % Find which columns are sampled (non-zero energy)
    col_energy = squeeze(sum(sum(abs(Mu), 1), 3));  % Ny x 1
    sampled_cols = find(col_energy > 0);
    Ry = round(mean(diff(sampled_cols(1:end))));
    if isempty(Ry) || Ry < 1; Ry = 1; end

    row_energy = squeeze(sum(sum(abs(Mu), 2), 3));  % Nx x 1
    sampled_rows = find(row_energy > 0);
    Rx = round(mean(diff(sampled_rows(1:end))));
    if isempty(Rx) || Rx < 1; Rx = 1; end

    fprintf('Detected undersampling: Rx=%d, Ry=%d\n', Rx, Ry);

    % Step 3: Fill missing k-space lines using GRAPPA kernel
    Mr = Mu;  % start with acquired data

    % Pad k-space for convolution
    pad_x = floor(kx/2);
    pad_y = floor(ky/2);
    Mu_pad = padarray(Mu, [pad_x, pad_y, 0], 0, 'both');

    % Loop over missing columns
    for y = 1:Ny
        if col_energy(y) > 0
            continue;  % skip sampled columns
        end

        % For each missing column y, find source patch
        for x = 1:Nx
            % Extract source patch around (x,y)
            x_range = x : x + kx - 1;
            y_range = y : y + ky - 1;

            patch = Mu_pad(x_range, y_range, :);

            % Apply kernel to predict missing k-space
            src = patch(:)';
            predicted = src * kernel;  % 1 x Nc

            Mr(x, y, :) = predicted;
        end
    end

    % Also fill missing rows if Rx > 1
    if Rx > 1
        row_energy_r = squeeze(sum(sum(abs(Mr), 2), 3));
        for x = 1:Nx
            if row_energy_r(x) > 0
                continue;
            end
            for y = 1:Ny
                x_range = x : x + kx - 1;
                y_range = y : y + ky - 1;
                patch = Mu_pad(x_range, y_range, :);
                src = patch(:)';
                predicted = src * kernel;
                Mr(x, y, :) = predicted;
            end
        end
    end

    % Step 4: Reconstruct images via inverse FFT
    imr = zeros(Nx, Ny, Nc);
    for c = 1:Nc
        imr(:,:,c) = fftshift(ifft2(ifftshift(Mr(:,:,c))));
    end
end

test_cases  = {[1,2],  [1,4],  [2,2]};
case_labels = {'Rx=1, Ry=2', 'Rx=1, Ry=4', 'Rx=2, Ry=2'};

calibx   = 24;
caliby   = 24;
lambda_g = 0.01;

[~, Mc] = imcalib(im, calibx, caliby);

figure('Units', 'normalized', 'Position', [0 0 1 0.6]);

for t = 1:3
    Rx = test_cases{t}(1);
    Ry = test_cases{t}(2);

    fprintf('\nCase %s:\n', case_labels{t});

    % Get undersampled k-space with calibration region
    [~, Mu] = undersamplecalib(im, Rx, Ry, calibx, caliby);

    % GRAPPA reconstruction
    [imr, Mr] = grappa(Mu, Mc, lambda_g);

    % SoS combination
    imr_sos = sqrt(sum(abs(imr).^2, 3));
    p95_grappa = prctile(imr_sos(:), 95);
    imr_norm = imr_sos / p95_grappa;

    % Error image
    error_img = abs(imr_norm - ref_img);

    % PSNR and SSIM
    psnr_g = psnr(imr_norm/ref_val, ref_img/ref_val);
    ssim_g = ssim(imr_norm/ref_val, ref_img/ref_val);

    fprintf('  PSNR=%.4f dB | SSIM=%.4f\n', psnr_g, ssim_g);

    % Display reconstruction
    subplot(2, 3, t);
    imshow(imr_norm, [0 ref_val]);
    colormap gray; colorbar;
    title(sprintf('%s\nPSNR=%.2f dB, SSIM=%.4f', ...
        case_labels{t}, psnr_g, ssim_g), 'FontSize', 8);

    % Display error
    subplot(2, 3, 3+t);
    imshow(error_img, [0 0.2*ref_val]);
    colormap gray; colorbar;
    title(sprintf('Error - %s', case_labels{t}), 'FontSize', 8);
end

annotation('textbox',[0.01 0.75 0.05 0.05],'String','Recon',...
    'EdgeColor','none','FontWeight','bold');
annotation('textbox',[0.01 0.25 0.05 0.05],'String','Error',...
    'EdgeColor','none','FontWeight','bold');
sgtitle('Part 3.3 - GRAPPA Reconstruction (SoS combination)');

%% Part 3.4 - GRAPPA Lambda Sweep

test_cases  = {[1,2],  [1,4],  [2,2]};
case_labels = {'Rx=1, Ry=2', 'Rx=1, Ry=4', 'Rx=2, Ry=2'};

calibx = 24;
caliby = 24;

% Lambda sweep
lambda_vals = logspace(-6, 2, 20);

[~, Mc] = imcalib(im, calibx, caliby);

best_lambda_g = zeros(1, 3);
best_psnr_g   = zeros(1, 3);
best_ssim_g   = zeros(1, 3);
best_imr      = cell(1, 3);

figure('Units', 'normalized', 'Position', [0 0 0.6 0.5]);

for t = 1:3
    Rx = test_cases{t}(1);
    Ry = test_cases{t}(2);

    [~, Mu] = undersamplecalib(im, Rx, Ry, calibx, caliby);

    psnr_sweep = zeros(1, length(lambda_vals));
    ssim_sweep = zeros(1, length(lambda_vals));

    fprintf('Case %s: sweeping lambda...\n', case_labels{t});

    for k = 1:length(lambda_vals)
        lam = lambda_vals(k);

        [imr, ~] = grappa(Mu, Mc, lam);

        % SoS combination
        imr_sos  = sqrt(sum(abs(imr).^2, 3));
        p95_g    = prctile(imr_sos(:), 95);
        imr_norm = imr_sos / p95_g;

        psnr_sweep(k) = psnr(imr_norm/ref_val, ref_img/ref_val);
        ssim_sweep(k) = ssim(imr_norm/ref_val, ref_img/ref_val);

        fprintf('  lambda=%.2e | PSNR=%.2f dB | SSIM=%.4f\n', ...
            lam, psnr_sweep(k), ssim_sweep(k));
    end

    % Best lambda by PSNR
    [best_psnr_g(t), idx] = max(psnr_sweep);
    best_lambda_g(t) = lambda_vals(idx);
    best_ssim_g(t)   = ssim_sweep(idx);

    % Store best reconstruction
    [imr_best, ~]  = grappa(Mu, Mc, best_lambda_g(t));
    imr_sos_best   = sqrt(sum(abs(imr_best).^2, 3));
    p95_best       = prctile(imr_sos_best(:), 95);
    best_imr{t}    = imr_sos_best / p95_best;

    fprintf('>> Best for %s: lambda=%.4e | PSNR=%.4f dB | SSIM=%.4f\n\n', ...
        case_labels{t}, best_lambda_g(t), best_psnr_g(t), best_ssim_g(t));

    % Plot PSNR vs lambda
    subplot(1, 3, t);
    semilogx(lambda_vals, psnr_sweep, 'b-o', 'LineWidth', 1.5, ...
        'MarkerSize', 4); hold on;
    xline(best_lambda_g(t), 'r--', 'LineWidth', 1.5);
    hold off;
    xlabel('\lambda'); ylabel('PSNR (dB)');
    title(sprintf('%s\nBest \\lambda=%.2e, PSNR=%.2f dB', ...
        case_labels{t}, best_lambda_g(t), best_psnr_g(t)), 'FontSize', 8);
    grid on;
end

sgtitle('Part 3.4 - GRAPPA PSNR vs \lambda');

figure('Units', 'normalized', 'Position', [0 0 1 0.6]);

for t = 1:3
    im_best   = best_imr{t};
    error_img = abs(im_best - ref_img);

    subplot(2, 3, t);
    imshow(im_best, [0 ref_val]);
    colormap gray; colorbar;
    title(sprintf('%s, \\lambda=%.2e\nPSNR=%.2f dB, SSIM=%.4f', ...
        case_labels{t}, best_lambda_g(t), best_psnr_g(t), best_ssim_g(t)), ...
        'FontSize', 8);

    subplot(2, 3, 3+t);
    imshow(error_img, [0 0.2*ref_val]);
    colormap gray; colorbar;
    title(sprintf('Error - %s', case_labels{t}), 'FontSize', 8);
end

annotation('textbox',[0.01 0.75 0.05 0.05],'String','Recon',...
    'EdgeColor','none','FontWeight','bold');
annotation('textbox',[0.01 0.25 0.05 0.05],'String','Error',...
    'EdgeColor','none','FontWeight','bold');
sgtitle('Part 3.4 - GRAPPA with Optimal Lambda');

fprintf('\n=== Summary: Best Lambda per Case (GRAPPA) ===\n');
fprintf('%-15s | %-12s | %-10s | %-10s\n', ...
    'Case', 'Lambda', 'PSNR (dB)', 'SSIM');
fprintf('%s\n', repmat('-', 1, 55));
for t = 1:3
    fprintf('%-15s | %-12.4e | %-10.4f | %-10.4f\n', ...
        case_labels{t}, best_lambda_g(t), best_psnr_g(t), best_ssim_g(t));
end

%% Part 3.5 - GRAPPA Calibration Region Size Sweep

test_cases  = {[1,2],  [1,4],  [2,2]};
case_labels = {'Rx=1, Ry=2', 'Rx=1, Ry=4', 'Rx=2, Ry=2'};
lambda_g    = 0.01;  % fixed lambda from Part 3.4

% Calibration sizes to sweep (square: calibx = caliby)
calib_sizes = [8, 12, 16, 20, 24, 32, 40, 48];

best_calib  = zeros(1, 3);
best_psnr_c = zeros(1, 3);
best_ssim_c = zeros(1, 3);
best_imr_c  = cell(1, 3);

% Store all PSNR curves for plotting
all_psnr = zeros(3, length(calib_sizes));
all_ssim = zeros(3, length(calib_sizes));

for t = 1:3
    Rx = test_cases{t}(1);
    Ry = test_cases{t}(2);

    fprintf('Case %s: sweeping calibration size...\n', case_labels{t});

    for k = 1:length(calib_sizes)
        cs = calib_sizes(k);

        % Get calibration data and undersampled k-space
        [~, Mc_k]  = imcalib(im, cs, cs);
        [~, Mu_k]  = undersamplecalib(im, Rx, Ry, cs, cs);

        % GRAPPA reconstruction
        [imr, ~]   = grappa(Mu_k, Mc_k, lambda_g);

        % SoS combination and normalization
        imr_sos    = sqrt(sum(abs(imr).^2, 3));
        p95_g      = prctile(imr_sos(:), 95);
        imr_norm   = imr_sos / p95_g;

        % PSNR and SSIM
        all_psnr(t,k) = psnr(imr_norm/ref_val, ref_img/ref_val);
        all_ssim(t,k) = ssim(imr_norm/ref_val, ref_img/ref_val);

        fprintf('  calib=%dx%d | PSNR=%.2f dB | SSIM=%.4f\n', ...
            cs, cs, all_psnr(t,k), all_ssim(t,k));
    end

    % Best calibration size by PSNR
    [best_psnr_c(t), idx] = max(all_psnr(t,:));
    best_calib(t)  = calib_sizes(idx);
    best_ssim_c(t) = all_ssim(t, idx);

    % Store best reconstruction
    [~, Mc_best]  = imcalib(im, best_calib(t), best_calib(t));
    [~, Mu_best]  = undersamplecalib(im, Rx, Ry, best_calib(t), best_calib(t));
    [imr_best, ~] = grappa(Mu_best, Mc_best, lambda_g);
    imr_sos_best  = sqrt(sum(abs(imr_best).^2, 3));
    best_imr_c{t} = imr_sos_best / prctile(imr_sos_best(:), 95);

    fprintf('>> Best for %s: calib=%dx%d | PSNR=%.4f dB | SSIM=%.4f\n\n', ...
        case_labels{t}, best_calib(t), best_calib(t), ...
        best_psnr_c(t), best_ssim_c(t));
end

figure('Units', 'normalized', 'Position', [0 0 0.7 0.45]);

colors = {'b', 'r', 'g'};
for t = 1:3
    plot(calib_sizes, all_psnr(t,:), [colors{t} '-o'], ...
        'LineWidth', 2, 'MarkerSize', 6, ...
        'DisplayName', case_labels{t}); hold on;
    xline(best_calib(t), ['--' colors{t}], 'LineWidth', 1.5, ...
        'HandleVisibility', 'off');
end
hold off;
xlabel('Calibration Region Size (N×N)');
ylabel('PSNR (dB)');
title('Part 3.5 - GRAPPA PSNR vs Calibration Region Size');
legend('Location', 'southeast');
grid on;
xticks(calib_sizes);

figure('Units', 'normalized', 'Position', [0 0 1 0.6]);

for t = 1:3
    im_best   = best_imr_c{t};
    error_img = abs(im_best - ref_img);

    subplot(2, 3, t);
    imshow(im_best, [0 ref_val]);
    colormap gray; colorbar;
    title(sprintf('%s, calib=%dx%d\nPSNR=%.2f dB, SSIM=%.4f', ...
        case_labels{t}, best_calib(t), best_calib(t), ...
        best_psnr_c(t), best_ssim_c(t)), 'FontSize', 8);

    subplot(2, 3, 3+t);
    imshow(error_img, [0 0.2*ref_val]);
    colormap gray; colorbar;
    title(sprintf('Error - %s', case_labels{t}), 'FontSize', 8);
end

annotation('textbox',[0.01 0.75 0.05 0.05],'String','Recon',...
    'EdgeColor','none','FontWeight','bold');
annotation('textbox',[0.01 0.25 0.05 0.05],'String','Error',...
    'EdgeColor','none','FontWeight','bold');
sgtitle('Part 3.5 - GRAPPA with Optimal Calibration Size');

fprintf('\n=== Summary: Best Calibration Size per Case ===\n');
fprintf('%-15s | %-12s | %-10s | %-10s\n', ...
    'Case', 'Calib Size', 'PSNR (dB)', 'SSIM');
fprintf('%s\n', repmat('-', 1, 55));
for t = 1:3
    fprintf('%-15s | %-12s | %-10.4f | %-10.4f\n', ...
        case_labels{t}, sprintf('%dx%d', best_calib(t), best_calib(t)), ...
        best_psnr_c(t), best_ssim_c(t));
end

