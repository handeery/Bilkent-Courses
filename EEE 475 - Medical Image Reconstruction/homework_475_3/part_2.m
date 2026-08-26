%% Task 2.1 - SVD of System Matrix S

% Compute compact SVD
[U, Sigma, V] = svd(S, 'econ');

% Verify sizes
fprintf('Size of U: %d x %d\n', size(U,1), size(U,2));
fprintf('Size of Sigma: %d x %d\n', size(Sigma,1), size(Sigma,2));
fprintf('Size of V: %d x %d\n', size(V,1), size(V,2));

% Extract diagonal singular values
sigma_vals = diag(Sigma);

% Plot singular values
figure;
plot(sigma_vals);
title('Singular Values of S');
xlabel('Index');
ylabel('Singular Value');
grid on;

% Also plot in log scale for better visualization
figure;
semilogy(sigma_vals);
title('Singular Values of S (Log Scale)');
xlabel('Index');
ylabel('Singular Value (log scale)');
grid on;

% Compute condition number manually
sigma_max = sigma_vals(1);       % largest singular value
sigma_min = sigma_vals(end);     % smallest singular value
cond_manual = sigma_max / sigma_min;
fprintf('Condition number (manual): %.4e\n', cond_manual);

% Verify using built-in cond function
cond_builtin = cond(S);
fprintf('Condition number (built-in): %.4e\n', cond_builtin);

%% Task 2.2 - Image Reconstruction via Moore-Penrose Pseudo-inverse (SVD)

% Compute pseudo-inverse solution: c = V * Sigma^-1 * U' * u
sigma_vals = diag(Sigma);
Sigma_inv = diag(1 ./ sigma_vals);
c = V * Sigma_inv * (U' * u);

% Reshape into 2D image
ima = reshape(c, 50, 50);

% Set negative values to zero
ima(ima < 0) = 0;

% Display reconstructed image
figure;
subplot(1,2,1);
imshow(ima, [0 max_val]);
colormap gray; colorbar;
title('SVD Reconstruction');
axis image;

% Compute and display error image
error_img = abs(ima - phantom_ref);

subplot(1,2,2);
imshow(error_img, [0 0.5*max_val]);
colormap gray; colorbar;
title('Error Image');
axis image;

sgtitle('SVD Reconstruction vs Error');

% Compute PSNR and SSIM
ima_norm = ima / max_val;
ref_norm = phantom_ref / max_val;

psnr_val = psnr(ima_norm, ref_norm);
ssim_val = ssim(ima_norm, ref_norm);

fprintf('PSNR: %.4f dB\n', psnr_val);
fprintf('SSIM: %.4f\n', ssim_val);

%%
% Task 2.3 - Truncated SVD

% Extract singular values
sigma_vals = diag(Sigma);

% Find M such that sigma(1)/sigma(M) ≈ 100
target_cond = 100;
sigma_max = sigma_vals(1);
M = sum(sigma_vals >= sigma_max / target_cond);
fprintf('M (number of kept singular values): %d\n', M);
fprintf('Achieved condition number: %.4f\n', sigma_max / sigma_vals(M));

% Truncate U, Sigma, V
U_trunc = U(:, 1:M);
Sigma_trunc = Sigma(1:M, 1:M);
V_trunc = V(:, 1:M);

fprintf('Size of U_trunc: %d x %d\n', size(U_trunc,1), size(U_trunc,2));
fprintf('Size of Sigma_trunc: %d x %d\n', size(Sigma_trunc,1), size(Sigma_trunc,2));
fprintf('Size of V_trunc: %d x %d\n', size(V_trunc,1), size(V_trunc,2));

% Compute reconstruction
sigma_trunc_vals = diag(Sigma_trunc);
Sigma_trunc_inv = diag(1 ./ sigma_trunc_vals);
c = V_trunc * Sigma_trunc_inv * (U_trunc' * u);

% Reshape into 2D image
ima_tsvd = reshape(c, 50, 50);

% Set negative values to zero
ima_tsvd(ima_tsvd < 0) = 0;

% Display reconstructed image and error
figure;
subplot(1,3,1);
imshow(phantom_ref, [0 max_val]);
colormap gray; colorbar;
title('Reference Phantom');
axis image;

subplot(1,3,2);
imshow(ima_tsvd, [0 max_val]);
colormap gray; colorbar;
title(sprintf('TSVD Reconstruction (M=%d)', M));
axis image;

subplot(1,3,3);
error_tsvd = abs(ima_tsvd - phantom_ref);
imshow(error_tsvd, [0 0.5*max_val]);
colormap gray; colorbar;
title('Error Image');
axis image;

sgtitle('Truncated SVD Reconstruction');

% Compute PSNR and SSIM
ima_tsvd_norm = ima_tsvd / max_val;
ref_norm = phantom_ref / max_val;

psnr_tsvd = psnr(ima_tsvd_norm, ref_norm);
ssim_tsvd = ssim(ima_tsvd_norm, ref_norm);

fprintf('PSNR: %.4f dB\n', psnr_tsvd);
fprintf('SSIM: %.4f\n', ssim_tsvd);

%%
% Task 2.4 - Truncated SVD for different condition numbers (single figure)

condition_numbers = [1, 5, 10, 100, 1e3, 1e4, 1e5, 1e6, 1e7];
n_conds = length(condition_numbers);

sigma_vals = diag(Sigma);
sigma_max = sigma_vals(1);

psnr_vals = zeros(1, n_conds);
ssim_vals = zeros(1, n_conds);
M_vals = zeros(1, n_conds);

ref_norm = phantom_ref / max_val;

% Single figure for all reconstructions
% Layout: 2 rows per condition (recon + error), but use 3 rows x 9 cols
% Better layout: 9 conditions x 2 images = 18 subplots -> 3 rows x 6 cols per type
% Clearest layout: 2 rows (recon on top, error on bottom) x 9 columns
figure('Units','normalized','Position',[0 0 1 0.6]);

for idx = 1:n_conds
    target_cond = condition_numbers(idx);
    
    % Find M for this condition number
    M = sum(sigma_vals >= sigma_max / target_cond);
    if M < 1, M = 1; end
    M_vals(idx) = M;
    
    % Truncate
    U_trunc = U(:, 1:M);
    Sigma_trunc_inv = diag(1 ./ sigma_vals(1:M));
    V_trunc = V(:, 1:M);
    
    % Reconstruct
    c = V_trunc * Sigma_trunc_inv * (U_trunc' * u);
    ima = reshape(c, 50, 50);
    ima(ima < 0) = 0;
    
    % Compute metrics
    ima_norm = ima / max_val;
    psnr_vals(idx) = psnr(ima_norm, ref_norm);
    ssim_vals(idx) = ssim(ima_norm, ref_norm);
    
    % Top row: reconstructed images
    subplot(2, n_conds, idx);
    imshow(ima, [0 max_val]);
    colormap gray;
    title(sprintf('Cond=10^{%d}\nM=%d', round(log10(max(target_cond,1))), M), 'FontSize', 7);
    axis image;
    
    % Bottom row: error images
    subplot(2, n_conds, n_conds + idx);
    error_img = abs(ima - phantom_ref);
    imshow(error_img, [0 0.5*max_val]);
    colormap gray;
    title(sprintf('PSNR=%.1f\nSSIM=%.2f', psnr_vals(idx), ssim_vals(idx)), 'FontSize', 7);
    axis image;
    
    fprintf('Cond: %.0e | M: %4d | PSNR: %.4f dB | SSIM: %.4f\n', ...
        target_cond, M, psnr_vals(idx), ssim_vals(idx));
end

% Add row labels
annotation('textbox',[0.01 0.75 0.05 0.05],'String','Recon','EdgeColor','none','FontWeight','bold');
annotation('textbox',[0.01 0.25 0.05 0.05],'String','Error','EdgeColor','none','FontWeight','bold');
sgtitle('Truncated SVD: Reconstructions and Error Images for Different Condition Numbers');

% Plot PSNR and SSIM vs condition number
figure;
subplot(2,1,1);
semilogx(condition_numbers, psnr_vals, 'b-o', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Condition Number (log scale)');
ylabel('PSNR (dB)');
title('PSNR vs Condition Number (TSVD)');
grid on;

subplot(2,1,2);
semilogx(condition_numbers, ssim_vals, 'r-o', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Condition Number (log scale)');
ylabel('SSIM');
title('SSIM vs Condition Number (TSVD)');
grid on;
sgtitle('Image Quality vs Condition Number');

% Find best condition number
[~, best_psnr_idx] = max(psnr_vals);
[~, best_ssim_idx] = max(ssim_vals);
fprintf('\nBest PSNR: %.4f dB at Cond = %.0e (M=%d)\n', ...
    psnr_vals(best_psnr_idx), condition_numbers(best_psnr_idx), M_vals(best_psnr_idx));
fprintf('Best SSIM: %.4f at Cond = %.0e (M=%d)\n', ...
    ssim_vals(best_ssim_idx), condition_numbers(best_ssim_idx), M_vals(best_ssim_idx));

%% Task 2.5 - Filtered SVD

sigma_vals = diag(Sigma);

% Test multiple lambda values
lambdas = [1e5, 1e8, 1e10, 1e12, 1e14, 1e16, 1e18, 1e20, 1e22];
n_lambdas = length(lambdas);

psnr_fsvd = zeros(1, n_lambdas);
ssim_fsvd = zeros(1, n_lambdas);

ref_norm = phantom_ref / max_val;

% Single figure for all reconstructions
figure('Units','normalized','Position',[0 0 1 0.6]);

for idx = 1:n_lambdas
    lambda = lambdas(idx);
    
    % Compute filter values y_i = sigma_i^2 / (sigma_i^2 + lambda)
    y = sigma_vals.^2 ./ (sigma_vals.^2 + lambda);
    
    % Compute filtered singular values sigma_i' = sigma_i / y_i
    sigma_filtered = sigma_vals ./ y;
    
    % Compute Sigma_tilde inverse: 1/sigma_i' = y_i / sigma_i
    Sigma_tilde_inv = diag(y ./ sigma_vals);
    
    % Reconstruct: c = V * Sigma_tilde^-1 * U' * u
    c = V * Sigma_tilde_inv * (U' * u);
    
    % Reshape and clip
    ima = reshape(c, 50, 50);
    ima(ima < 0) = 0;
    
    % Compute metrics
    ima_norm = ima / max_val;
    psnr_fsvd(idx) = psnr(ima_norm, ref_norm);
    ssim_fsvd(idx) = ssim(ima_norm, ref_norm);
    
    % Top row: reconstructed images
    subplot(2, n_lambdas, idx);
    imshow(ima, [0 max_val]);
    colormap gray;
    title(sprintf('\\lambda=10^{%d}', round(log10(lambda))), 'FontSize', 7);
    axis image;
    
    % Bottom row: error images
    subplot(2, n_lambdas, n_lambdas + idx);
    error_img = abs(ima - phantom_ref);
    imshow(error_img, [0 0.5*max_val]);
    colormap gray;
    title(sprintf('PSNR=%.1f\nSSIM=%.2f', psnr_fsvd(idx), ssim_fsvd(idx)), 'FontSize', 7);
    axis image;
    
    fprintf('Lambda: %.0e | PSNR: %.4f dB | SSIM: %.4f\n', lambda, psnr_fsvd(idx), ssim_fsvd(idx));
end

annotation('textbox',[0.01 0.75 0.05 0.05],'String','Recon','EdgeColor','none','FontWeight','bold');
annotation('textbox',[0.01 0.25 0.05 0.05],'String','Error','EdgeColor','none','FontWeight','bold');
sgtitle('Filtered SVD: Reconstructions and Error Images for Different \lambda Values');

% Plot PSNR and SSIM vs lambda
figure;
subplot(2,1,1);
semilogx(lambdas, psnr_fsvd, 'b-o', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('\lambda (log scale)');
ylabel('PSNR (dB)');
title('PSNR vs \lambda (Filtered SVD)');
grid on;

subplot(2,1,2);
semilogx(lambdas, ssim_fsvd, 'r-o', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('\lambda (log scale)');
ylabel('SSIM');
title('SSIM vs \lambda (Filtered SVD)');
grid on;
sgtitle('Image Quality vs \lambda (Filtered SVD)');

% Find best lambda
[~, best_psnr_idx] = max(psnr_fsvd);
[~, best_ssim_idx] = max(ssim_fsvd);
fprintf('\nBest PSNR: %.4f dB at lambda = %.0e\n', psnr_fsvd(best_psnr_idx), lambdas(best_psnr_idx));
fprintf('Best SSIM: %.4f at lambda = %.0e\n', ssim_fsvd(best_ssim_idx), lambdas(best_ssim_idx));

%% Task 2.6 - Filtered SVD with lambda = sigma_1 * sigma_N

sigma_vals = diag(Sigma);
sigma_1 = sigma_vals(1);      % largest singular value
sigma_N = sigma_vals(end);    % smallest singular value

% Choose lambda
lambda = sigma_1 * sigma_N;
fprintf('sigma_1 = %.4e\n', sigma_1);
fprintf('sigma_N = %.4e\n', sigma_N);
fprintf('lambda = sigma_1 * sigma_N = %.4e\n', lambda);

% Compute filter values
y = sigma_vals.^2 ./ (sigma_vals.^2 + lambda);

% Compute filtered singular values sigma_i' = sigma_i / y_i
sigma_filtered = sigma_vals ./ y;

% Plot original vs filtered singular values
figure;
subplot(1,2,1);
semilogy(sigma_vals, 'b-', 'LineWidth', 2);
hold on;
semilogy(sigma_filtered, 'r--', 'LineWidth', 2);
xlabel('Index');
ylabel('Singular Value (log scale)');
title('Original vs Filtered Singular Values');
legend('\sigma_i (original)', "\sigma_i' (filtered)", 'Location', 'northeast');
grid on;

subplot(1,2,2);
semilogy(sigma_vals, 'b-', 'LineWidth', 2);
hold on;
semilogy(sigma_filtered, 'r--', 'LineWidth', 2);
xline(find(sigma_vals.^2 >= lambda, 1, 'last'), 'g--', 'LineWidth', 1.5, ...
    'Label', '\sigma_i^2 = \lambda');
xlabel('Index');
ylabel('Singular Value (log scale)');
title('Zoom: Transition Region');
legend('\sigma_i (original)', "\sigma_i' (filtered)", 'Location', 'northeast');
xlim([0 2500]);
grid on;

sgtitle(sprintf('Singular Value Filtering with \\lambda = \\sigma_1 \\cdot \\sigma_N = %.2e', lambda));

% Reconstruct
Sigma_tilde_inv = diag(y ./ sigma_vals);
c = V * Sigma_tilde_inv * (U' * u);
ima_fsvd = reshape(c, 50, 50);
ima_fsvd(ima_fsvd < 0) = 0;

% Display reconstructed image and error
figure;
subplot(1,3,1);
imshow(phantom_ref, [0 max_val]);
colormap gray; colorbar;
title('Reference Phantom');
axis image;

subplot(1,3,2);
imshow(ima_fsvd, [0 max_val]);
colormap gray; colorbar;
title(sprintf('Filtered SVD\n\\lambda=\\sigma_1\\cdot\\sigma_N=%.2e', lambda));
axis image;

subplot(1,3,3);
error_img = abs(ima_fsvd - phantom_ref);
imshow(error_img, [0 0.5*max_val]);
colormap gray; colorbar;
title('Error Image');
axis image;

sgtitle('Filtered SVD Reconstruction (\lambda = \sigma_1 \cdot \sigma_N)');

% Compute PSNR and SSIM
ima_fsvd_norm = ima_fsvd / max_val;
ref_norm = phantom_ref / max_val;
psnr_fsvd = psnr(ima_fsvd_norm, ref_norm);
ssim_fsvd = ssim(ima_fsvd_norm, ref_norm);

fprintf('PSNR: %.4f dB\n', psnr_fsvd);
fprintf('SSIM: %.4f\n', ssim_fsvd);

% Show how filter behaves at large and small singular values
fprintf('\nFilter value at largest singular value (i=1):   y_1 = %.6f\n', y(1));
fprintf('Filter value at smallest singular value (i=N): y_N = %.6f\n', y(end));

%% Task 2.7 - Filtered SVD: Finding optimal lambda

% Test range of lambda values (powers of 10)
lambdas = 10.^(8:0.5:16);
n_lambdas = length(lambdas);

psnr_vals = zeros(1, n_lambdas);
ssim_vals = zeros(1, n_lambdas);
ref_norm = phantom_ref / max_val;
sigma_vals = diag(Sigma);

for idx = 1:n_lambdas
    lambda = lambdas(idx);
    y = sigma_vals.^2 ./ (sigma_vals.^2 + lambda);
    Sigma_tilde_inv = diag(y ./ sigma_vals);
    c = V * Sigma_tilde_inv * (U' * u);
    ima = reshape(c, 50, 50);
    ima(ima < 0) = 0;
    ima_norm = ima / max_val;
    psnr_vals(idx) = psnr(ima_norm, ref_norm);
    ssim_vals(idx) = ssim(ima_norm, ref_norm);
    fprintf('Lambda: %.2e | PSNR: %.4f dB | SSIM: %.4f\n', lambda, psnr_vals(idx), ssim_vals(idx));
end

% Plot PSNR and SSIM vs lambda
figure;
subplot(2,1,1);
semilogx(lambdas, psnr_vals, 'b-o', 'LineWidth', 2, 'MarkerSize', 6);
xlabel('\lambda (log scale)');
ylabel('PSNR (dB)');
title('PSNR vs \lambda (Filtered SVD)');
grid on;

subplot(2,1,2);
semilogx(lambdas, ssim_vals, 'r-o', 'LineWidth', 2, 'MarkerSize', 6);
xlabel('\lambda (log scale)');
ylabel('SSIM');
title('SSIM vs \lambda (Filtered SVD)');
grid on;
sgtitle('Image Quality vs \lambda - Finding Optimal \lambda^*');

% Find best lambda
[~, best_psnr_idx] = max(psnr_vals);
[~, best_ssim_idx] = max(ssim_vals);
lambda_star = lambdas(best_psnr_idx);
fprintf('\nBest PSNR: %.4f dB at lambda* = %.2e\n', psnr_vals(best_psnr_idx), lambda_star);
fprintf('Best SSIM: %.4f at lambda = %.2e\n', ssim_vals(best_ssim_idx), lambdas(best_ssim_idx));

%% Display results for lambda*/10, lambda*, 10*lambda*
test_lambdas = [lambda_star/10, lambda_star, lambda_star*10];
test_labels  = {'\lambda^*/10', '\lambda^*', '10\lambda^*'};

figure('Units','normalized','Position',[0 0 1 0.6]);

for idx = 1:3
    lambda = test_lambdas(idx);
    y = sigma_vals.^2 ./ (sigma_vals.^2 + lambda);
    Sigma_tilde_inv = diag(y ./ sigma_vals);
    c = V * Sigma_tilde_inv * (U' * u);
    ima = reshape(c, 50, 50);
    ima(ima < 0) = 0;
    ima_norm = ima / max_val;
    p = psnr(ima_norm, ref_norm);
    s = ssim(ima_norm, ref_norm);

    % Reconstruction
    subplot(2, 3, idx);
    imshow(ima, [0 max_val]);
    colormap gray; colorbar;
    title(sprintf('%s = %.2e\nPSNR=%.2f dB, SSIM=%.4f', ...
        test_labels{idx}, lambda, p, s), 'FontSize', 8);
    axis image;

    % Error image
    subplot(2, 3, 3 + idx);
    error_img = abs(ima - phantom_ref);
    imshow(error_img, [0 0.5*max_val]);
    colormap gray; colorbar;
    title(sprintf('Error - %s', test_labels{idx}), 'FontSize', 8);
    axis image;

    fprintf('%s = %.2e | PSNR: %.4f dB | SSIM: %.4f\n', ...
        test_labels{idx}, lambda, p, s);
end

% Add row labels
annotation('textbox',[0.01 0.75 0.05 0.05],'String','Recon', ...
    'EdgeColor','none','FontWeight','bold');
annotation('textbox',[0.01 0.25 0.05 0.05],'String','Error', ...
    'EdgeColor','none','FontWeight','bold');
sgtitle(sprintf('Filtered SVD: \\lambda^*/10, \\lambda^*, 10\\lambda^* (\\lambda^* = %.2e)', ...
    lambda_star));

% Display results for lambda*/10, lambda*, 10*lambda*
lambda_star = 3.16e+11;
test_lambdas = [lambda_star/10, lambda_star, lambda_star*10];
test_labels  = {'\lambda^*/10', '\lambda^*', '10\lambda^*'};

figure('Units','normalized','Position',[0 0 0.9 0.6]);

for idx = 1:3
    lambda = test_lambdas(idx);
    y = sigma_vals.^2 ./ (sigma_vals.^2 + lambda);
    Sigma_tilde_inv = diag(y ./ sigma_vals);
    c = V * Sigma_tilde_inv * (U' * u);
    ima = reshape(c, 50, 50);
    ima(ima < 0) = 0;
    ima_norm = ima / max_val;
    p = psnr(ima_norm, ref_norm);
    s = ssim(ima_norm, ref_norm);

    % Reconstruction
    subplot(2, 3, idx);
    imshow(ima, [0 max_val]);
    colormap gray; colorbar;
    title(sprintf('%s = %.2e\nPSNR=%.2f dB, SSIM=%.4f', ...
        test_labels{idx}, lambda, p, s), 'FontSize', 8);
    axis image;

    % Error image
    subplot(2, 3, 3 + idx);
    error_img = abs(ima - phantom_ref);
    imshow(error_img, [0 0.5*max_val]);
    colormap gray; colorbar;
    title(sprintf('Error - %s', test_labels{idx}), 'FontSize', 8);
    axis image;

    fprintf('%s = %.2e | PSNR: %.4f dB | SSIM: %.4f\n', ...
        test_labels{idx}, lambda, p, s);
end

annotation('textbox',[0.01 0.75 0.05 0.05],'String','Recon', ...
    'EdgeColor','none','FontWeight','bold');
annotation('textbox',[0.01 0.25 0.05 0.05],'String','Error', ...
    'EdgeColor','none','FontWeight','bold');
sgtitle(sprintf('Filtered SVD: \\lambda^*/10, \\lambda^*, 10\\lambda^*  (\\lambda^* = %.2e)', ...
    lambda_star));

%% Task 2.8 - L-curve method

lambdas = 10.^(6:0.25:20);
n_lambdas = length(lambdas);

residual_norms = zeros(1, n_lambdas);
solution_norms = zeros(1, n_lambdas);

sigma_vals = diag(Sigma);
ref_norm = phantom_ref / max_val;

for idx = 1:n_lambdas
    lambda = lambdas(idx);
    y = sigma_vals.^2 ./ (sigma_vals.^2 + lambda);
    Sigma_tilde_inv = diag(y ./ sigma_vals);
    c = V * Sigma_tilde_inv * (U' * u);

    % Residual norm ||Sc - u||
    residual_norms(idx) = norm(S*c - u);

    % Solution norm ||c||
    solution_norms(idx) = norm(c);
end

% Plot L-curve
figure;
loglog(residual_norms, solution_norms, 'b-o', 'LineWidth', 2, 'MarkerSize', 5);
xlabel('Residual Norm ||Sc - u||');
ylabel('Solution Norm ||c||');
grid on;

% Annotate lambda values on the curve
label_indices = 1:4:n_lambdas;
for i = label_indices
    text(residual_norms(i), solution_norms(i), ...
        sprintf('  10^{%.1f}', log10(lambdas(i))), ...
        'FontSize', 7, 'Color', 'red');
end

% Mark automatic corner (for reference)
log_res = log10(residual_norms);
log_sol = log10(solution_norms);
dx = gradient(log_res);
dy = gradient(log_sol);
ddx = gradient(dx);
ddy = gradient(dy);
curvature = abs(dx.*ddy - dy.*ddx) ./ (dx.^2 + dy.^2).^1.5;
[~, corner_idx] = max(curvature);
lambda_auto = lambdas(corner_idx);

% Manual corner based on visual inspection
lambda_lcurve_manual = 3.16e+11;

% Mark both on L-curve
hold on;
plot(residual_norms(corner_idx), solution_norms(corner_idx), ...
    'r*', 'MarkerSize', 15, 'LineWidth', 2);
manual_idx = find(abs(lambdas - lambda_lcurve_manual) == ...
    min(abs(lambdas - lambda_lcurve_manual)));
plot(residual_norms(manual_idx), solution_norms(manual_idx), ...
    'g*', 'MarkerSize', 15, 'LineWidth', 2);
legend('L-curve', ...
    sprintf('Auto corner: \\lambda = %.2e', lambda_auto), ...
    sprintf('Manual corner: \\lambda = %.2e', lambda_lcurve_manual), ...
    'Location', 'best');
title(sprintf('L-Curve'));

fprintf('Automatic L-curve corner at lambda = %.2e\n', lambda_auto);
fprintf('Manual L-curve corner at lambda = %.2e\n', lambda_lcurve_manual);

% Reconstruct for both automatic and manual lambda*
lambdas_to_test = [lambda_auto, lambda_lcurve_manual];
labels = {'Auto L-curve \lambda^*', 'Manual L-curve \lambda^*'};

figure('Units','normalized','Position',[0 0 0.9 0.6]);

for idx = 1:2
    lambda = lambdas_to_test(idx);
    y = sigma_vals.^2 ./ (sigma_vals.^2 + lambda);
    Sigma_tilde_inv = diag(y ./ sigma_vals);
    c = V * Sigma_tilde_inv * (U' * u);
    ima_lc = reshape(c, 50, 50);
    ima_lc(ima_lc < 0) = 0;

    ima_lc_norm = ima_lc / max_val;
    p = psnr(ima_lc_norm, ref_norm);
    s = ssim(ima_lc_norm, ref_norm);

    fprintf('%s: lambda = %.2e | PSNR: %.4f dB | SSIM: %.4f\n', ...
        labels{idx}, lambda, p, s);

    % Reconstruction
    subplot(2, 3, (idx-1)*3 + 1);
    imshow(phantom_ref, [0 max_val]);
    colormap gray; colorbar;
    title('Reference Phantom');
    axis image;

    subplot(2, 3, (idx-1)*3 + 2);
    imshow(ima_lc, [0 max_val]);
    colormap gray; colorbar;
    title(sprintf('%s\n\\lambda=%.2e\nPSNR=%.2f dB, SSIM=%.4f', ...
        labels{idx}, lambda, p, s), 'FontSize', 8);
    axis image;

    subplot(2, 3, (idx-1)*3 + 3);
    error_img = abs(ima_lc - phantom_ref);
    imshow(error_img, [0 0.5*max_val]);
    colormap gray; colorbar;
    title('Error Image');
    axis image;
end

sgtitle('L-curve Reconstruction: Automatic vs Manual Corner');