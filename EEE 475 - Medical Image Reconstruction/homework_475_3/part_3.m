% Task 3.1 - Row-norm Thresholding

% Compute norm of each row of S
row_norms = vecnorm(S, 2, 2);  % 2-norm along dimension 2 (columns)

% Define max_norm
max_norm = max(row_norms);
fprintf('max_norm = %.4e\n', max_norm);

% Plot row norms
figure;
plot(row_norms, 'b-', 'LineWidth', 1);
hold on;
yline(max_norm/10, 'r--', 'LineWidth', 2, 'Label', 'max\_norm/10 threshold');
xlabel('Row Index');
ylabel('Row Norm');
title('Row Norms of System Matrix S');
legend('Row norms', 'Threshold (max\_norm/10)', 'Location', 'best');
grid on;

% Find rows above threshold
threshold = max_norm / 10;
keep_rows = row_norms >= threshold;
fprintf('Number of rows kept: %d out of %d\n', sum(keep_rows), size(S,1));
fprintf('Number of rows discarded: %d\n', sum(~keep_rows));

% Discard rows below threshold from S and u
S_thresh = S(keep_rows, :);
u_thresh = u(keep_rows);

% Display sizes
fprintf('Size of S after thresholding: %d x %d\n', size(S_thresh,1), size(S_thresh,2));
fprintf('Size of u after thresholding: %d x %d\n', size(u_thresh,1), size(u_thresh,2));

%% Task 3.2 - Standard Kaczmarz Method

% Use thresholded matrices from Task 3.1
% S_thresh: 44 x 2500, u_thresh: 44 x 1

n_iter = 10;
n_rows = size(S_thresh, 1);   % 44
n_cols = size(S_thresh, 2);   % 2500

% Precompute row norms squared for efficiency
row_norms_thresh = vecnorm(S_thresh, 2, 2);
row_norms_sq = row_norms_thresh.^2;

% Initialize c as zeros
c = zeros(n_cols, 1);

ref_norm = phantom_ref / max_val;

psnr_kacz = zeros(1, n_iter);
ssim_kacz = zeros(1, n_iter);

% Single figure for all iterations
figure('Units','normalized','Position',[0 0 1 0.6]);

for iter = 1:n_iter
    % Randomize row order for this iteration
    row_order = randperm(n_rows);
    
    % Sub-iterations: cycle through all rows in random order
    for k = 1:n_rows
        i = row_order(k);
        s_i = S_thresh(i, :)';          % column vector
        u_i = u_thresh(i);              % scalar
        
        % Kaczmarz update
        c = c + ((u_i - s_i' * c) / row_norms_sq(i)) * s_i;
    end
    
    % Set negative values to zero after each full iteration
    c(c < 0) = 0;
    
    % Reshape to image
    ima = reshape(c, 50, 50);
    
    % Compute metrics
    ima_norm = ima / max_val;
    psnr_kacz(iter) = psnr(ima_norm, ref_norm);
    ssim_kacz(iter) = ssim(ima_norm, ref_norm);
    
    fprintf('Iter %2d | PSNR: %.4f dB | SSIM: %.4f\n', ...
        iter, psnr_kacz(iter), ssim_kacz(iter));
    
    % Top row: reconstructed images
    subplot(2, n_iter, iter);
    imshow(ima, [0 max_val]);
    colormap gray;
    title(sprintf('Iter %d', iter), 'FontSize', 7);
    axis image;
    
    % Bottom row: error images
    subplot(2, n_iter, n_iter + iter);
    error_img = abs(ima - phantom_ref);
    imshow(error_img, [0 0.5*max_val]);
    colormap gray;
    title(sprintf('PSNR=%.1f\nSSIM=%.2f', psnr_kacz(iter), ssim_kacz(iter)), 'FontSize', 7);
    axis image;
end

annotation('textbox',[0.01 0.75 0.05 0.05],'String','Recon', ...
    'EdgeColor','none','FontWeight','bold');
annotation('textbox',[0.01 0.25 0.05 0.05],'String','Error', ...
    'EdgeColor','none','FontWeight','bold');
sgtitle('Standard Kaczmarz: Iterations 1-10');

% Plot PSNR and SSIM vs iteration
figure;
subplot(2,1,1);
plot(1:n_iter, psnr_kacz, 'b-o', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Iteration');
ylabel('PSNR (dB)');
title('PSNR vs Iteration (Standard Kaczmarz)');
xticks(1:n_iter);
grid on;

subplot(2,1,2);
plot(1:n_iter, ssim_kacz, 'r-o', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Iteration');
ylabel('SSIM');
title('SSIM vs Iteration (Standard Kaczmarz)');
xticks(1:n_iter);
grid on;

sgtitle('Image Quality vs Iteration (Standard Kaczmarz)');

%% Task 3.3 - Standard Kaczmarz with different row-norm thresholds

threshold_factors = [1e-4, 1e-3, 1e-2, 1e-1];
n_thresh = length(threshold_factors);
n_iter = 10;

% Recompute full row norms from original S
row_norms_full = vecnorm(S, 2, 2);
max_norm = max(row_norms_full);

psnr_thresh = zeros(1, n_thresh);
ssim_thresh = zeros(1, n_thresh);
rows_kept   = zeros(1, n_thresh);

ref_norm = phantom_ref / max_val;

% Single figure for all threshold results
figure('Units','normalized','Position',[0 0 1 0.6]);

for t = 1:n_thresh
    factor = threshold_factors(t);
    threshold = max_norm * factor;

    % Threshold rows
    keep_rows = row_norms_full >= threshold;
    S_t = S(keep_rows, :);
    u_t = u(keep_rows);
    rows_kept(t) = sum(keep_rows);

    fprintf('Threshold: max_norm x %.0e | Rows kept: %d\n', factor, rows_kept(t));

    % Precompute row norms squared
    rn_sq = vecnorm(S_t, 2, 2).^2;

    n_rows_t = size(S_t, 1);

    % Initialize c
    c = zeros(2500, 1);

    % Run 10 iterations
    for iter = 1:n_iter
        row_order = randperm(n_rows_t);
        for k = 1:n_rows_t
            i = row_order(k);
            s_i = S_t(i, :)';
            u_i = u_t(i);
            c = c + ((u_i - s_i'*c) / rn_sq(i)) * s_i;
        end
        % Set negatives to zero after each full iteration
        c(c < 0) = 0;
    end

    % Reshape to image
    ima = reshape(c, 50, 50);

    % Compute metrics
    ima_norm = ima / max_val;
    psnr_thresh(t) = psnr(ima_norm, ref_norm);
    ssim_thresh(t) = ssim(ima_norm, ref_norm);

    fprintf('  Iter 10 | PSNR: %.4f dB | SSIM: %.4f\n', psnr_thresh(t), ssim_thresh(t));

    % Top row: reconstructed images
    subplot(2, n_thresh, t);
    imshow(ima, [0 max_val]);
    colormap gray;
    title(sprintf('Thresh=%.0e\nRows kept=%d', factor, rows_kept(t)), 'FontSize', 8);
    axis image;

    % Bottom row: error images
    subplot(2, n_thresh, n_thresh + t);
    error_img = abs(ima - phantom_ref);
    imshow(error_img, [0 0.5*max_val]);
    colormap gray;
    title(sprintf('PSNR=%.2f dB\nSSIM=%.4f', psnr_thresh(t), ssim_thresh(t)), 'FontSize', 8);
    axis image;
end

annotation('textbox',[0.01 0.75 0.05 0.05],'String','Recon', ...
    'EdgeColor','none','FontWeight','bold');
annotation('textbox',[0.01 0.25 0.05 0.05],'String','Error', ...
    'EdgeColor','none','FontWeight','bold');
sgtitle('Standard Kaczmarz (Iter 10): Different Row-Norm Thresholds');

% Plot PSNR and SSIM vs threshold factor
figure;
subplot(2,1,1);
semilogx(threshold_factors, psnr_thresh, 'b-o', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Threshold Factor (log scale)');
ylabel('PSNR (dB)');
title('PSNR vs Row-Norm Threshold (Iter 10)');
xticks(threshold_factors);
xticklabels({'10^{-4}','10^{-3}','10^{-2}','10^{-1}'});
grid on;

subplot(2,1,2);
semilogx(threshold_factors, ssim_thresh, 'r-o', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Threshold Factor (log scale)');
ylabel('SSIM');
title('SSIM vs Row-Norm Threshold (Iter 10)');
xticks(threshold_factors);
xticklabels({'10^{-4}','10^{-3}','10^{-2}','10^{-1}'});
grid on;

sgtitle('Image Quality vs Row-Norm Threshold (Standard Kaczmarz, Iter 10)');

% Summary
fprintf('\nSummary at Iteration 10:\n');
fprintf('%-20s %-15s %-15s %-15s\n','Threshold','Rows Kept','PSNR (dB)','SSIM');
for t = 1:n_thresh
    fprintf('max_norm x %.0e   %-15d %-15.4f %-15.4f\n', ...
        threshold_factors(t), rows_kept(t), psnr_thresh(t), ssim_thresh(t));
end

%% Task 3.4 - Regularized Kaczmarz Method
% No row-norm thresholding - use full S and u

% Lambda = sigma_1 * sigma_N (from Task 2.6)
sigma_vals = diag(Sigma);
lambda = sigma_vals(1) * sigma_vals(end);
fprintf('lambda = sigma_1 * sigma_N = %.4e\n', lambda);

n_iter = 10;
n_rows = size(S, 1);   % 3264 (full, no thresholding)
n_cols = size(S, 2);   % 2500

% Precompute row norms squared + lambda for regularized update
row_norms_sq_full = vecnorm(S, 2, 2).^2;
denom = row_norms_sq_full + lambda;  % ||s_i||^2 + lambda

% Initialize c as zeros
c = zeros(n_cols, 1);

ref_norm = phantom_ref / max_val;

psnr_rkacz = zeros(1, n_iter);
ssim_rkacz = zeros(1, n_iter);

% Single figure for all iterations
figure('Units','normalized','Position',[0 0 1 0.6]);

for iter = 1:n_iter
    % Randomize row order
    row_order = randperm(n_rows);

    % Sub-iterations
    for k = 1:n_rows
        i = row_order(k);
        s_i = S(i, :)';
        u_i = u(i);

        % Regularized Kaczmarz update:
        % c = c + (u_i - s_i'*c) / (||s_i||^2 + lambda) * s_i
        c = c + ((u_i - s_i'*c) / denom(i)) * s_i;
    end

    % Set negative values to zero after each full iteration
    c(c < 0) = 0;

    % Reshape to image
    ima = reshape(c, 50, 50);

    % Compute metrics
    ima_norm = ima / max_val;
    psnr_rkacz(iter) = psnr(ima_norm, ref_norm);
    ssim_rkacz(iter) = ssim(ima_norm, ref_norm);

    fprintf('Iter %2d | PSNR: %.4f dB | SSIM: %.4f\n', ...
        iter, psnr_rkacz(iter), ssim_rkacz(iter));

    % Top row: reconstructed images
    subplot(2, n_iter, iter);
    imshow(ima, [0 max_val]);
    colormap gray;
    title(sprintf('Iter %d', iter), 'FontSize', 7);
    axis image;

    % Bottom row: error images
    subplot(2, n_iter, n_iter + iter);
    error_img = abs(ima - phantom_ref);
    imshow(error_img, [0 0.5*max_val]);
    colormap gray;
    title(sprintf('PSNR=%.1f\nSSIM=%.2f', psnr_rkacz(iter), ssim_rkacz(iter)), 'FontSize', 7);
    axis image;
end

annotation('textbox',[0.01 0.75 0.05 0.05],'String','Recon', ...
    'EdgeColor','none','FontWeight','bold');
annotation('textbox',[0.01 0.25 0.05 0.05],'String','Error', ...
    'EdgeColor','none','FontWeight','bold');
sgtitle(sprintf('Regularized Kaczmarz (\\lambda=%.2e): Iterations 1-10', lambda));

% Plot PSNR and SSIM vs iteration
figure;
subplot(2,1,1);
plot(1:n_iter, psnr_rkacz, 'b-o', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Iteration');
ylabel('PSNR (dB)');
title(sprintf('PSNR vs Iteration (Regularized Kaczmarz, \\lambda=%.2e)', lambda));
xticks(1:n_iter);
grid on;

subplot(2,1,2);
plot(1:n_iter, ssim_rkacz, 'r-o', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Iteration');
ylabel('SSIM');
title(sprintf('SSIM vs Iteration (Regularized Kaczmarz, \\lambda=%.2e)', lambda));
xticks(1:n_iter);
grid on;

sgtitle(sprintf('Image Quality vs Iteration (Regularized Kaczmarz, \\lambda=%.2e)', lambda));

%% Task 3.5 - Regularized Kaczmarz with lambda = sigma_1^2 * lambda_rel

sigma_vals = diag(Sigma);
sigma_1_sq = sigma_vals(1)^2;

lambda_rel_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2];
n_lambda = length(lambda_rel_values);
n_iter = 10;
n_rows = size(S, 1);   % 3264 (no thresholding)
n_cols = size(S, 2);   % 2500

ref_norm = phantom_ref / max_val;

% Precompute row norms squared (same for all lambda)
row_norms_sq_full = vecnorm(S, 2, 2).^2;

psnr_rel = zeros(1, n_lambda);
ssim_rel = zeros(1, n_lambda);

% Single figure for all lambda_rel results
figure('Units','normalized','Position',[0 0 1 0.6]);

for t = 1:n_lambda
    lambda_rel = lambda_rel_values(t);
    lambda = sigma_1_sq * lambda_rel;
    fprintf('lambda_rel = %.0e | lambda = %.4e\n', lambda_rel, lambda);

    % Denominator: ||s_i||^2 + lambda
    denom = row_norms_sq_full + lambda;

    % Initialize c
    c = zeros(n_cols, 1);

    % Run 10 iterations
    for iter = 1:n_iter
        row_order = randperm(n_rows);
        for k = 1:n_rows
            i = row_order(k);
            s_i = S(i, :)';
            u_i = u(i);
            c = c + ((u_i - s_i'*c) / denom(i)) * s_i;
        end
        % Set negatives to zero after each full iteration
        c(c < 0) = 0;
    end

    % Reshape to image
    ima = reshape(c, 50, 50);

    % Compute metrics
    ima_norm = ima / max_val;
    psnr_rel(t) = psnr(ima_norm, ref_norm);
    ssim_rel(t) = ssim(ima_norm, ref_norm);

    fprintf('  lambda_rel=%.0e | PSNR: %.4f dB | SSIM: %.4f\n', ...
        lambda_rel, psnr_rel(t), ssim_rel(t));

    % Top row: reconstructed images
    subplot(2, n_lambda, t);
    imshow(ima, [0 max_val]);
    colormap gray;
    title(sprintf('\\lambda_{rel}=10^{%d}\n\\lambda=%.1e', ...
        round(log10(lambda_rel)), lambda), 'FontSize', 7);
    axis image;

    % Bottom row: error images
    subplot(2, n_lambda, n_lambda + t);
    error_img = abs(ima - phantom_ref);
    imshow(error_img, [0 0.5*max_val]);
    colormap gray;
    title(sprintf('PSNR=%.2f dB\nSSIM=%.4f', psnr_rel(t), ssim_rel(t)), 'FontSize', 7);
    axis image;
end

annotation('textbox',[0.01 0.75 0.05 0.05],'String','Recon', ...
    'EdgeColor','none','FontWeight','bold');
annotation('textbox',[0.01 0.25 0.05 0.05],'String','Error', ...
    'EdgeColor','none','FontWeight','bold');
sgtitle('Regularized Kaczmarz (Iter 10): Different \lambda_{rel} Values');

% Plot PSNR and SSIM vs lambda_rel
figure;
subplot(2,1,1);
semilogx(lambda_rel_values, psnr_rel, 'b-o', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('\lambda_{rel} (log scale)');
ylabel('PSNR (dB)');
title('PSNR vs \lambda_{rel} (Regularized Kaczmarz, Iter 10)');
xticks(lambda_rel_values);
xticklabels({'10^{-6}','10^{-5}','10^{-4}','10^{-3}','10^{-2}'});
grid on;

subplot(2,1,2);
semilogx(lambda_rel_values, ssim_rel, 'r-o', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('\lambda_{rel} (log scale)');
ylabel('SSIM');
title('SSIM vs \lambda_{rel} (Regularized Kaczmarz, Iter 10)');
xticks(lambda_rel_values);
xticklabels({'10^{-6}','10^{-5}','10^{-4}','10^{-3}','10^{-2}'});
grid on;

sgtitle('Image Quality vs \lambda_{rel} (Regularized Kaczmarz, Iter 10)');

% Summary
fprintf('\nSummary at Iteration 10:\n');
fprintf('%-15s %-20s %-15s %-15s\n','lambda_rel','lambda','PSNR (dB)','SSIM');
for t = 1:n_lambda
    fprintf('%-15.0e %-20.4e %-15.4f %-15.4f\n', ...
        lambda_rel_values(t), sigma_1_sq*lambda_rel_values(t), psnr_rel(t), ssim_rel(t));
end

%% Task 3.6 - Effects of Measurement Noise

% Add white Gaussian noise to u
noise_std = max_norm / 1000;
fprintf('Noise standard deviation: %.4e\n', noise_std);

rng(42); % for reproducibility
u_noisy = u + noise_std * randn(size(u));

fprintf('Original u norm: %.4e\n', norm(u));
fprintf('Noise norm: %.4e\n', norm(noise_std * randn(size(u))));
fprintf('SNR: %.2f dB\n', 20*log10(norm(u) / (noise_std * sqrt(length(u)))));

% Row-norm thresholds (same as Task 3.3)
threshold_factors = [1e-4, 1e-3, 1e-2, 1e-1];
n_thresh = length(threshold_factors);
n_iter = 10;

row_norms_full = vecnorm(S, 2, 2);
max_norm_val = max(row_norms_full);

psnr_noisy = zeros(1, n_thresh);
ssim_noisy = zeros(1, n_thresh);
rows_kept  = zeros(1, n_thresh);

ref_norm = phantom_ref / max_val;

% Single figure for all threshold results
figure('Units','normalized','Position',[0 0 1 0.6]);

for t = 1:n_thresh
    factor = threshold_factors(t);
    threshold = max_norm_val * factor;

    % Threshold rows (apply same thresholding to noisy u)
    keep_rows = row_norms_full >= threshold;
    S_t = S(keep_rows, :);
    u_t = u_noisy(keep_rows);   % use noisy u
    rows_kept(t) = sum(keep_rows);

    fprintf('Threshold: max_norm x %.0e | Rows kept: %d\n', factor, rows_kept(t));

    % Precompute row norms squared
    rn_sq = vecnorm(S_t, 2, 2).^2;
    n_rows_t = size(S_t, 1);

    % Initialize c
    c = zeros(2500, 1);

    % Run 10 iterations
    for iter = 1:n_iter
        row_order = randperm(n_rows_t);
        for k = 1:n_rows_t
            i = row_order(k);
            s_i = S_t(i, :)';
            u_i = u_t(i);
            c = c + ((u_i - s_i'*c) / rn_sq(i)) * s_i;
        end
        c(c < 0) = 0;
    end

    % Reshape to image
    ima = reshape(c, 50, 50);

    % Compute metrics
    ima_norm = ima / max_val;
    psnr_noisy(t) = psnr(ima_norm, ref_norm);
    ssim_noisy(t) = ssim(ima_norm, ref_norm);

    fprintf('  Iter 10 | PSNR: %.4f dB | SSIM: %.4f\n', psnr_noisy(t), ssim_noisy(t));

    % Top row: reconstructed images
    subplot(2, n_thresh, t);
    imshow(ima, [0 max_val]);
    colormap gray;
    title(sprintf('Thresh=%.0e\nRows=%d', factor, rows_kept(t)), 'FontSize', 8);
    axis image;

    % Bottom row: error images
    subplot(2, n_thresh, n_thresh + t);
    error_img = abs(ima - phantom_ref);
    imshow(error_img, [0 0.5*max_val]);
    colormap gray;
    title(sprintf('PSNR=%.2f dB\nSSIM=%.4f', psnr_noisy(t), ssim_noisy(t)), 'FontSize', 8);
    axis image;
end

annotation('textbox',[0.01 0.75 0.05 0.05],'String','Recon', ...
    'EdgeColor','none','FontWeight','bold');
annotation('textbox',[0.01 0.25 0.05 0.05],'String','Error', ...
    'EdgeColor','none','FontWeight','bold');
sgtitle('Standard Kaczmarz with Noisy u (Iter 10): Different Row-Norm Thresholds');

% Plot PSNR and SSIM vs threshold — noisy vs noiseless comparison
% Noiseless results from Task 3.3
psnr_noiseless = [28.9529, 27.9286, 23.0736, 14.2973];
ssim_noiseless = [0.9165,  0.9311,  0.7436,  0.4487];

figure;
subplot(2,1,1);
semilogx(threshold_factors, psnr_noiseless, 'b-o', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
semilogx(threshold_factors, psnr_noisy, 'r-s', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Threshold Factor (log scale)');
ylabel('PSNR (dB)');
title('PSNR vs Row-Norm Threshold (Iter 10)');
xticks(threshold_factors);
xticklabels({'10^{-4}','10^{-3}','10^{-2}','10^{-1}'});
legend('Noiseless', 'Noisy', 'Location', 'best');
grid on;

subplot(2,1,2);
semilogx(threshold_factors, ssim_noiseless, 'b-o', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
semilogx(threshold_factors, ssim_noisy, 'r-s', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Threshold Factor (log scale)');
ylabel('SSIM');
title('SSIM vs Row-Norm Threshold (Iter 10)');
xticks(threshold_factors);
xticklabels({'10^{-4}','10^{-3}','10^{-2}','10^{-1}'});
legend('Noiseless', 'Noisy', 'Location', 'best');
grid on;

sgtitle('Image Quality vs Row-Norm Threshold: Noiseless vs Noisy');

% Summary
fprintf('\nComparison Summary at Iteration 10:\n');
fprintf('%-15s %-10s %-15s %-15s %-15s %-15s\n', ...
    'Threshold','Rows','PSNR_clean','PSNR_noisy','SSIM_clean','SSIM_noisy');
for t = 1:n_thresh
    fprintf('%.0e           %-10d %-15.4f %-15.4f %-15.4f %-15.4f\n', ...
        threshold_factors(t), rows_kept(t), ...
        psnr_noiseless(t), psnr_noisy(t), ...
        ssim_noiseless(t), ssim_noisy(t));
end


%% Task 3.7 - Regularized Kaczmarz with noise, lambda = sigma_1^2 * lambda_rel

% Use u_noisy from Task 3.6
sigma_vals = diag(Sigma);
sigma_1_sq = sigma_vals(1)^2;

lambda_rel_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2];
n_lambda = length(lambda_rel_values);
n_iter = 10;
n_rows = size(S, 1);   % 3264 (no thresholding)
n_cols = size(S, 2);   % 2500

ref_norm = phantom_ref / max_val;

% Precompute row norms squared (same for all lambda)
row_norms_sq_full = vecnorm(S, 2, 2).^2;

psnr_rel_noisy = zeros(1, n_lambda);
ssim_rel_noisy = zeros(1, n_lambda);

% Noiseless results from Task 3.5 for comparison
psnr_rel_clean = [28.3106, 27.8860, 26.8717, 22.8877, 17.9772];
ssim_rel_clean = [0.9578,  0.9405,  0.9097,  0.8777,  0.7118];

% Single figure for all lambda_rel results
figure('Units','normalized','Position',[0 0 1 0.6]);

for t = 1:n_lambda
    lambda_rel = lambda_rel_values(t);
    lambda = sigma_1_sq * lambda_rel;
    fprintf('lambda_rel = %.0e | lambda = %.4e\n', lambda_rel, lambda);

    % Denominator: ||s_i||^2 + lambda
    denom = row_norms_sq_full + lambda;

    % Initialize c
    c = zeros(n_cols, 1);

    % Run 10 iterations using noisy u
    for iter = 1:n_iter
        row_order = randperm(n_rows);
        for k = 1:n_rows
            i = row_order(k);
            s_i = S(i, :)';
            u_i = u_noisy(i);    % noisy measurement
            c = c + ((u_i - s_i'*c) / denom(i)) * s_i;
        end
        c(c < 0) = 0;
    end

    % Reshape to image
    ima = reshape(c, 50, 50);

    % Compute metrics
    ima_norm = ima / max_val;
    psnr_rel_noisy(t) = psnr(ima_norm, ref_norm);
    ssim_rel_noisy(t) = ssim(ima_norm, ref_norm);

    fprintf('  lambda_rel=%.0e | PSNR: %.4f dB | SSIM: %.4f\n', ...
        lambda_rel, psnr_rel_noisy(t), ssim_rel_noisy(t));

    % Top row: reconstructed images
    subplot(2, n_lambda, t);
    imshow(ima, [0 max_val]);
    colormap gray;
    title(sprintf('\\lambda_{rel}=10^{%d}\n\\lambda=%.1e', ...
        round(log10(lambda_rel)), lambda), 'FontSize', 7);
    axis image;

    % Bottom row: error images
    subplot(2, n_lambda, n_lambda + t);
    error_img = abs(ima - phantom_ref);
    imshow(error_img, [0 0.5*max_val]);
    colormap gray;
    title(sprintf('PSNR=%.2f dB\nSSIM=%.4f', ...
        psnr_rel_noisy(t), ssim_rel_noisy(t)), 'FontSize', 7);
    axis image;
end

annotation('textbox',[0.01 0.75 0.05 0.05],'String','Recon', ...
    'EdgeColor','none','FontWeight','bold');
annotation('textbox',[0.01 0.25 0.05 0.05],'String','Error', ...
    'EdgeColor','none','FontWeight','bold');
sgtitle('Regularized Kaczmarz with Noisy u (Iter 10): Different \lambda_{rel} Values');

% Plot PSNR and SSIM vs lambda_rel — noisy vs noiseless comparison
figure;
subplot(2,1,1);
semilogx(lambda_rel_values, psnr_rel_clean, 'b-o', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
semilogx(lambda_rel_values, psnr_rel_noisy, 'r-s', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('\lambda_{rel} (log scale)');
ylabel('PSNR (dB)');
title('PSNR vs \lambda_{rel} (Regularized Kaczmarz, Iter 10)');
xticks(lambda_rel_values);
xticklabels({'10^{-6}','10^{-5}','10^{-4}','10^{-3}','10^{-2}'});
legend('Noiseless', 'Noisy', 'Location', 'best');
grid on;

subplot(2,1,2);
semilogx(lambda_rel_values, ssim_rel_clean, 'b-o', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
semilogx(lambda_rel_values, ssim_rel_noisy, 'r-s', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('\lambda_{rel} (log scale)');
ylabel('SSIM');
title('SSIM vs \lambda_{rel} (Regularized Kaczmarz, Iter 10)');
xticks(lambda_rel_values);
xticklabels({'10^{-6}','10^{-5}','10^{-4}','10^{-3}','10^{-2}'});
legend('Noiseless', 'Noisy', 'Location', 'best');
grid on;

sgtitle('Image Quality vs \lambda_{rel}: Noiseless vs Noisy (Regularized Kaczmarz)');

% Summary
fprintf('\nComparison Summary at Iteration 10:\n');
fprintf('%-15s %-20s %-15s %-15s %-15s %-15s\n', ...
    'lambda_rel','lambda','PSNR_clean','PSNR_noisy','SSIM_clean','SSIM_noisy');
for t = 1:n_lambda
    fprintf('%-15.0e %-20.4e %-15.4f %-15.4f %-15.4f %-15.4f\n', ...
        lambda_rel_values(t), sigma_1_sq*lambda_rel_values(t), ...
        psnr_rel_clean(t), psnr_rel_noisy(t), ...
        ssim_rel_clean(t), ssim_rel_noisy(t));
end