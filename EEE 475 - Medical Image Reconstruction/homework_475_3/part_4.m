% Task 4.1 - OpenMPI Dataset: System Matrix and SVD

% Load OpenMPI data with different variable names to avoid overwriting
load('/Users/handeeryilmaz/Downloads/medikal/homework_475_3/su_openmpi.mat');
% This loads S and u — but these are the OpenMPI versions now

fprintf('Size of S (OpenMPI): %d x %d\n', size(S,1), size(S,2));
fprintf('Size of u (OpenMPI): %d x %d\n', size(u,1), size(u,2));

% Rename to avoid confusion with simulated S and u
S_mpi = S;
u_mpi = u;

% Restore simulated S and u (reload if needed)
load('/Users/handeeryilmaz/Downloads/medikal/homework_475_3/system_matrix_S.mat');     % loads S
load('/Users/handeeryilmaz/Downloads/medikal/homework_475_3/measurement_vector_u.mat'); % loads u

fprintf('Size of S_mpi: %d x %d\n', size(S_mpi,1), size(S_mpi,2));
fprintf('Size of u_mpi: %d x %d\n', size(u_mpi,1), size(u_mpi,2));

% Plot three different columns of OpenMPI system matrix
locations = [100, 5000, 25000];

figure;
for i = 1:3
    subplot(1, 3, i);
    plot(S_mpi(:, locations(i)));
    title(sprintf('System Matrix Column\nGrid Location %d', locations(i)));
    xlabel('Frequency Index');
    ylabel('Amplitude');
    grid on;
end
sgtitle('OpenMPI System Matrix Columns at Different Grid Locations');

% Compute compact SVD (may take a while)
fprintf('Computing SVD of S_mpi (this may take a while)...\n');
tic;
[U_mpi, Sigma_mpi, V_mpi] = svd(S_mpi, 'econ');
toc;

fprintf('Size of U_mpi:     %d x %d\n', size(U_mpi,1),     size(U_mpi,2));
fprintf('Size of Sigma_mpi: %d x %d\n', size(Sigma_mpi,1), size(Sigma_mpi,2));
fprintf('Size of V_mpi:     %d x %d\n', size(V_mpi,1),     size(V_mpi,2));

% Extract singular values
sigma_mpi_vals = diag(Sigma_mpi);

% Plot singular values
figure;
subplot(1,2,1);
plot(sigma_mpi_vals, 'b-', 'LineWidth', 1.5);
title('Singular Values of S (OpenMPI)');
xlabel('Index');
ylabel('Singular Value');
grid on;

subplot(1,2,2);
semilogy(sigma_mpi_vals, 'b-', 'LineWidth', 1.5);
title('Singular Values of S (OpenMPI) - Log Scale');
xlabel('Index');
ylabel('Singular Value (log scale)');
grid on;

sgtitle('Singular Values of OpenMPI System Matrix');

% Compute condition number
sigma_max_mpi = sigma_mpi_vals(1);
sigma_min_mpi = sigma_mpi_vals(end);
cond_mpi = sigma_max_mpi / sigma_min_mpi;
fprintf('Largest singular value:  %.4e\n', sigma_max_mpi);
fprintf('Smallest singular value: %.4e\n', sigma_min_mpi);
fprintf('Condition number: %.4e\n', cond_mpi);

%% Task 4.2 - SVD of OpenMPI System Matrix (already computed in 4.1)

% Singular values already available as sigma_mpi_vals
% Just ensure they are extracted
sigma_mpi_vals = diag(Sigma_mpi);

% Plot singular values
figure;
subplot(1,2,1);
plot(sigma_mpi_vals, 'b-', 'LineWidth', 1.5);
title('Singular Values of S (OpenMPI)');
xlabel('Index');
ylabel('Singular Value');
grid on;

subplot(1,2,2);
semilogy(sigma_mpi_vals, 'b-', 'LineWidth', 1.5);
title('Singular Values of S (OpenMPI) - Log Scale');
xlabel('Index');
ylabel('Singular Value (log scale)');
grid on;

sgtitle('Singular Values of OpenMPI System Matrix');

% Condition number
sigma_max_mpi = sigma_mpi_vals(1);
sigma_min_mpi = sigma_mpi_vals(end);
cond_manual   = sigma_max_mpi / sigma_min_mpi;
cond_builtin  = cond(S_mpi);

fprintf('Largest singular value:          %.4e\n', sigma_max_mpi);
fprintf('Smallest singular value:         %.4e\n', sigma_min_mpi);
fprintf('Condition number (manual):       %.4e\n', cond_manual);
fprintf('Condition number (built-in):     %.4e\n', cond_builtin);

%% Task 4.3 - Regularized Kaczmarz on OpenMPI data

% Lambda = sigma_1 * sigma_N
sigma_max_mpi = sigma_mpi_vals(1);
sigma_min_mpi = sigma_mpi_vals(end);
lambda_mpi = sigma_max_mpi * sigma_min_mpi;
fprintf('sigma_1 = %.4e\n', sigma_max_mpi);
fprintf('sigma_N = %.4e\n', sigma_min_mpi);
fprintf('lambda = sigma_1 * sigma_N = %.4e\n', lambda_mpi);

n_iter = 10;
n_rows_mpi = size(S_mpi, 1);   % 3056
n_cols_mpi = size(S_mpi, 2);   % 50653

% Precompute row norms squared + lambda
row_norms_sq_mpi = vecnorm(S_mpi, 2, 2).^2;
denom_mpi = row_norms_sq_mpi + lambda_mpi;

% Initialize c
c_mpi = zeros(n_cols_mpi, 1);

fprintf('Running Regularized Kaczmarz for %d iterations...\n', n_iter);

for iter = 1:n_iter
    tic;
    row_order = randperm(n_rows_mpi);
    for k = 1:n_rows_mpi
        i = row_order(k);
        s_i = S_mpi(i, :)';
        u_i = u_mpi(i);
        c_mpi = c_mpi + ((u_i - s_i'*c_mpi) / denom_mpi(i)) * s_i;
    end
    % Set negatives to zero
    c_mpi(c_mpi < 0) = 0;
    elapsed = toc;
    fprintf('Iter %2d completed in %.2f seconds\n', iter, elapsed);
end

% Reshape to 3D volume
ima_mpi = reshape(c_mpi, 37, 37, 37);

% Display all 37 slices as montage
figure;
montage(reshape(ima_mpi, [37, 37, 1, 37]), 'displayRange', []);
colormap gray;
colorbar;
title(sprintf('OpenMPI Reconstruction - Regularized Kaczmarz (\\lambda=%.2e)\nAll 37 Slices', ...
    lambda_mpi));

% Also display a few individual slices for better inspection
figure;
slice_indices = [10, 15, 19, 22, 25, 28];
n_slices = length(slice_indices);
max_val_mpi = max(ima_mpi(:));

for i = 1:n_slices
    subplot(2, 3, i);
    imagesc(ima_mpi(:,:,slice_indices(i)));
    colormap gray; colorbar;
    axis image;
    title(sprintf('Slice %d', slice_indices(i)));
end
sgtitle(sprintf('OpenMPI Reconstruction - Selected Slices (\\lambda=%.2e)', lambda_mpi));

fprintf('Max intensity: %.4e\n', max_val_mpi);
fprintf('Reconstruction complete.\n');

%% Task 4.4 - Regularized Kaczmarz on OpenMPI with different lambda_rel values

sigma_1_sq_mpi = sigma_mpi_vals(1)^2;

lambda_rel_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2];
n_lambda = length(lambda_rel_values);
n_iter = 10;
n_rows_mpi = size(S_mpi, 1);   % 3056
n_cols_mpi = size(S_mpi, 2);   % 50653

% Precompute row norms squared (same for all lambda)
row_norms_sq_mpi = vecnorm(S_mpi, 2, 2).^2;

for t = 1:n_lambda
    lambda_rel = lambda_rel_values(t);
    lambda = sigma_1_sq_mpi * lambda_rel;
    fprintf('\nlambda_rel = %.0e | lambda = %.4e\n', lambda_rel, lambda);

    % Denominator
    denom_mpi = row_norms_sq_mpi + lambda;

    % Initialize c
    c_mpi = zeros(n_cols_mpi, 1);

    % Run 10 iterations
    for iter = 1:n_iter
        row_order = randperm(n_rows_mpi);
        for k = 1:n_rows_mpi
            i = row_order(k);
            s_i = S_mpi(i, :)';
            u_i = u_mpi(i);
            c_mpi = c_mpi + ((u_i - s_i'*c_mpi) / denom_mpi(i)) * s_i;
        end
        c_mpi(c_mpi < 0) = 0;
        fprintf('  Iter %2d done\n', iter);
    end

    % Reshape to 3D volume
    ima_mpi = reshape(c_mpi, 37, 37, 37);
    max_val_mpi = max(ima_mpi(:));
    fprintf('Max intensity: %.4e\n', max_val_mpi);

    % Display montage of all 37 slices
    figure;
    montage(reshape(ima_mpi, [37, 37, 1, 37]), 'displayRange', []);
    colormap gray;
    title(sprintf('OpenMPI Recon - \\lambda_{rel}=10^{%d}, \\lambda=%.2e (Iter 10)', ...
        round(log10(lambda_rel)), lambda), 'FontSize', 10);
end

fprintf('\nAll reconstructions complete.\n');