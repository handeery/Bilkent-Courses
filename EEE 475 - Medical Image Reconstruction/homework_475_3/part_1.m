% Task 1.1 - Viewing Calibration Data
% Three different grid locations
locations = [5, 100, 500];

figure;
for coil = 1:2
    for i = 1:3
        subplot(2, 3, (coil-1)*3 + i);
        plot(abs(SM(coil, :, locations(i))));
        title(sprintf('Coil #%d, Grid Location %d', coil, locations(i)));
        xlabel('Frequency Index');
        ylabel('Magnitude');
        grid on;
    end
end

sgtitle('Magnitude Spectrum of Calibration Measurements');

%%
% Task 1.2 - Preparing the System Matrix

% Step 1: Discard second half of frequency-domain data (conjugate symmetry)
S = SM(:, 1:end/2, :);
% S is now 2x816x2500

% Step 2: Reshape to concatenate two coils along frequency axis
S = reshape(S, 2*816, 2500);
% S is now 1632x2500

% Step 3: Concatenate real and imaginary parts along frequency axis
S = [real(S); imag(S)];
% S is now 3264x2500, purely real-valued

% Verify size
disp(size(S));

% Plot three different columns of the system matrix
locations = [5, 100, 500];

figure;
for i = 1:3
    subplot(1, 3, i);
    plot(S(:, locations(i)));
    title(sprintf('System Matrix Column - Grid Location %d', locations(i)));
    xlabel('Frequency Index');
    ylabel('Amplitude');
    grid on;
end
sgtitle('System Matrix Columns at Different Grid Locations');

% Save the system matrix S
save('/Users/handeeryilmaz/Downloads/medikal/homework_475_3/system_matrix_S.mat', 'S');
disp('System matrix saved.');

% Check file size
fileInfo = dir('/Users/handeeryilmaz/Downloads/medikal/homework_475_3/system_matrix_S.mat');
fprintf('File size: %.2f MB\n', fileInfo.bytes / (1024^2));

%%
% Task 1.3 - Preparing the Measurement Vector

% Display the phantom
figure;
imagesc(phantom_ref);
colormap gray;
colorbar;
title('Reference Phantom (m_{ref})');
axis image;

% Define max_val
max_val = max(phantom_ref(:));
fprintf('max_val = %.4f\n', max_val);

% Prepare measurement vector u
u = meas(:, 1:end/2);
u = reshape(u, 2*816, 1);
u = [real(u); imag(u)];

% Verify size
fprintf('Size of u: %d x %d\n', size(u,1), size(u,2));

% Plot measurement vector
figure;
plot(u);
title('Measurement Vector u');
xlabel('Index');
ylabel('Amplitude');
grid on;

% Save measurement vector
save('/Users/handeeryilmaz/Downloads/medikal/homework_475_3/measurement_vector_u.mat', 'u');
disp('Measurement vector saved.');

% Check file size
fileInfo = dir('/Users/handeeryilmaz/Downloads/medikal/homework_475_3/measurement_vector_u.mat');
fprintf('File size: %.4f KB\n', fileInfo.bytes / 1024);