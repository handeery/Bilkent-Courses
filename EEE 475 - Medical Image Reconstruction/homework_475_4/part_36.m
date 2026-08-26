%% Part 3.6 Final Comparison



function [imu, Mu] = undersample(im, Rx, Ry)
    [Nx, Ny, Nc] = size(im);
    M = zeros(Nx, Ny, Nc);
    for c = 1:Nc
        M(:,:,c) = fftshift(fft2(ifftshift(im(:,:,c))));
    end
    row_indices = 1:Rx:Nx;
    col_indices = 1:Ry:Ny;
    Mu = zeros(Nx/Rx, Ny/Ry, Nc);
    imu = zeros(Nx/Rx, Ny/Ry, Nc);
    for c = 1:Nc
        Mu(:,:,c)  = M(row_indices, col_indices, c);
        imu(:,:,c) = fftshift(ifft2(ifftshift(Mu(:,:,c))));
    end
end

function im_sense = l2sense(imu, map, Rx, Ry, lambda)
    [Nxu, Nyu, Nc] = size(imu);
    Nx = Nxu * Rx; Ny = Nyu * Ry;
    im_sense = zeros(Nx, Ny);
    warning('off', 'MATLAB:rankDeficientMatrix');
    for x = 1:Nxu
        for y = 1:Nyu
            src_x = x:Nxu:Nx; src_y = y:Nyu:Ny;
            n_src = length(src_x) * length(src_y);
            S = zeros(Nc, n_src);
            idx = 1;
            for ix = 1:length(src_x)
                for iy = 1:length(src_y)
                    S(:,idx) = map(src_x(ix), src_y(iy), :);
                    idx = idx + 1;
                end
            end
            m_alias = squeeze(imu(x, y, :));
            c_vec = (S'*S + lambda*eye(n_src)) \ (S'*m_alias);
            idx = 1;
            for ix = 1:length(src_x)
                for iy = 1:length(src_y)
                    im_sense(src_x(ix), src_y(iy)) = c_vec(idx);
                    idx = idx + 1;
                end
            end
        end
    end
    warning('on', 'MATLAB:rankDeficientMatrix');
    im_sense = abs(im_sense);
    im_sense(isnan(im_sense)) = 0;
    im_sense(isinf(im_sense)) = 0;
end

function [imc, Mc] = imcalib(im, calibx, caliby)
    [Nx, Ny, Nc] = size(im);
    M = zeros(Nx, Ny, Nc);
    for c = 1:Nc
        M(:,:,c) = fftshift(fft2(ifftshift(im(:,:,c))));
    end
    cx = round(Nx/2); cy = round(Ny/2);
    x_calib = cx - floor(calibx/2) : cx + ceil(calibx/2) - 1;
    y_calib = cy - floor(caliby/2) : cy + ceil(caliby/2) - 1;
    Mc = M(x_calib, y_calib, :);
    imc = zeros(calibx, caliby, Nc);
    for c = 1:Nc
        imc(:,:,c) = fftshift(ifft2(ifftshift(Mc(:,:,c))));
    end
end

function [imu, Mu] = undersamplecalib(im, Rx, Ry, calibx, caliby)
    [Nx, Ny, Nc] = size(im);
    M = zeros(Nx, Ny, Nc);
    for c = 1:Nc
        M(:,:,c) = fftshift(fft2(ifftshift(im(:,:,c))));
    end
    cx = round(Nx/2); cy = round(Ny/2);
    x_calib = cx - floor(calibx/2) : cx + ceil(calibx/2) - 1;
    y_calib = cy - floor(caliby/2) : cy + ceil(caliby/2) - 1;
    mask = zeros(Nx, Ny);
    mask(1:Rx:Nx, 1:Ry:Ny) = 1;
    mask(x_calib, y_calib) = 1;
    Mu = zeros(Nx, Ny, Nc);
    for c = 1:Nc
        Mu(:,:,c) = M(:,:,c) .* mask;
    end
    row_indices = 1:Rx:Nx;
    col_indices = 1:Ry:Ny;
    Mu_sub = Mu(row_indices, col_indices, :);
    imu = zeros(Nx/Rx, Ny/Ry, Nc);
    for c = 1:Nc
        imu(:,:,c) = fftshift(ifft2(ifftshift(Mu_sub(:,:,c))));
    end
end

function kernel = calibrate(Mc, lambda)
    [calibx, caliby, Nc] = size(Mc);
    kx = 3; ky = 4;
    n_src_pts = kx * ky * Nc;
    n_patches = (calibx - kx + 1) * (caliby - ky + 1);
    A = zeros(n_patches, n_src_pts);
    B = zeros(n_patches, Nc);
    patch_idx = 1;
    for px = 1 : calibx - kx + 1
        for py = 1 : caliby - ky + 1
            patch = Mc(px:px+kx-1, py:py+ky-1, :);
            A(patch_idx, :) = patch(:)';
            cx = px + floor(kx/2);
            cy = py + floor(ky/2);
            B(patch_idx, :) = squeeze(Mc(cx, cy, :))';
            patch_idx = patch_idx + 1;
        end
    end
    kernel = (A'*A + lambda*eye(n_src_pts)) \ (A'*B);
end

function [imr, Mr] = grappa(Mu, Mc, lambda)
    [Nx, Ny, Nc] = size(Mu);
    kernel = calibrate(Mc, lambda);
    kx = 3; ky = 4;
    col_energy = squeeze(sum(sum(abs(Mu), 1), 3));
    Mu_pad = padarray(Mu, [floor(kx/2), floor(ky/2), 0], 0, 'both');
    Mr = Mu;
    for y = 1:Ny
        if col_energy(y) > 0; continue; end
        for x = 1:Nx
            patch = Mu_pad(x:x+kx-1, y:y+ky-1, :);
            Mr(x, y, :) = patch(:)' * kernel;
        end
    end
    row_energy = squeeze(sum(sum(abs(Mr), 2), 3));
    for x = 1:Nx
        if row_energy(x) > 0; continue; end
        for y = 1:Ny
            patch = Mu_pad(x:x+kx-1, y:y+ky-1, :);
            Mr(x, y, :) = patch(:)' * kernel;
        end
    end
    imr = zeros(Nx, Ny, Nc);
    for c = 1:Nc
        imr(:,:,c) = fftshift(ifft2(ifftshift(Mr(:,:,c))));
    end
end


test_cases  = {[1,2],  [1,4],  [2,2]};
case_labels = {'Rx=1, Ry=2', 'Rx=1, Ry=4', 'Rx=2, Ry=2'};

best_lambdas_sense  = [178, 100, 3.16e4];
best_lambdas_grappa = 0.01;

methods   = {'SENSE map1', 'SENSE map2', 'SoS (zf)', ...
             'GRAPPA 24x24', 'GRAPPA 48x48'};
n_methods = length(methods);
psnr_all  = zeros(n_methods, 3);
ssim_all  = zeros(n_methods, 3);
im_all    = cell(n_methods, 3);

for t = 1:3
    Rx    = test_cases{t}(1);
    Ry    = test_cases{t}(2);
    lam_s = best_lambdas_sense(t);

    fprintf('=== Case %s ===\n', case_labels{t});
    [imu, ~] = undersample(im, Rx, Ry);

    %--- SENSE map1 ---%
    im_raw = l2sense(imu, map1, Rx, Ry, lam_s);
    im_s1  = im_raw / prctile(im_raw(:), 95);
    psnr_all(1,t) = psnr(im_s1/ref_val, ref_img/ref_val);
    ssim_all(1,t) = ssim(im_s1/ref_val, ref_img/ref_val);
    im_all{1,t}   = im_s1;
    fprintf('  SENSE map1:   PSNR=%.4f | SSIM=%.4f\n', psnr_all(1,t), ssim_all(1,t));

    %--- SENSE map2 ---%
    im_raw = l2sense(imu, map2, Rx, Ry, lam_s);
    im_s2  = im_raw / prctile(im_raw(:), 95);
    psnr_all(2,t) = psnr(im_s2/ref_val, ref_img/ref_val);
    ssim_all(2,t) = ssim(im_s2/ref_val, ref_img/ref_val);
    im_all{2,t}   = im_s2;
    fprintf('  SENSE map2:   PSNR=%.4f | SSIM=%.4f\n', psnr_all(2,t), ssim_all(2,t));

    %--- SoS zero-filled ---%
    [Nxu, Nyu, Nc] = size(imu);
    Nx = Nxu*Rx; Ny = Nyu*Ry;
    im_zf = zeros(Nx, Ny, Nc);
    for c = 1:Nc
        Mzf = zeros(Nx, Ny);
        Mzf(1:Rx:Nx, 1:Ry:Ny) = fftshift(fft2(ifftshift(imu(:,:,c))));
        im_zf(:,:,c) = fftshift(ifft2(ifftshift(Mzf)));
    end
    sos_zf = sqrt(sum(abs(im_zf).^2, 3));
    im_sos = sos_zf / prctile(sos_zf(:), 95);
    psnr_all(3,t) = psnr(im_sos/ref_val, ref_img/ref_val);
    ssim_all(3,t) = ssim(im_sos/ref_val, ref_img/ref_val);
    im_all{3,t}   = im_sos;
    fprintf('  SoS (zf):     PSNR=%.4f | SSIM=%.4f\n', psnr_all(3,t), ssim_all(3,t));

    %--- GRAPPA 24x24 ---%
    [~, Mc24]  = imcalib(im, 24, 24);
    [~, Mu24]  = undersamplecalib(im, Rx, Ry, 24, 24);
    [imr24, ~] = grappa(Mu24, Mc24, best_lambdas_grappa);
    sos24      = sqrt(sum(abs(imr24).^2, 3));
    im_g24     = sos24 / prctile(sos24(:), 95);
    psnr_all(4,t) = psnr(im_g24/ref_val, ref_img/ref_val);
    ssim_all(4,t) = ssim(im_g24/ref_val, ref_img/ref_val);
    im_all{4,t}   = im_g24;
    fprintf('  GRAPPA 24x24: PSNR=%.4f | SSIM=%.4f\n', psnr_all(4,t), ssim_all(4,t));

    %--- GRAPPA 48x48 ---%
    [~, Mc48]  = imcalib(im, 48, 48);
    [~, Mu48]  = undersamplecalib(im, Rx, Ry, 48, 48);
    [imr48, ~] = grappa(Mu48, Mc48, best_lambdas_grappa);
    sos48      = sqrt(sum(abs(imr48).^2, 3));
    im_g48     = sos48 / prctile(sos48(:), 95);
    psnr_all(5,t) = psnr(im_g48/ref_val, ref_img/ref_val);
    ssim_all(5,t) = ssim(im_g48/ref_val, ref_img/ref_val);
    im_all{5,t}   = im_g48;
    fprintf('  GRAPPA 48x48: PSNR=%.4f | SSIM=%.4f\n', psnr_all(5,t), ssim_all(5,t));

    fprintf('\n');
end

figure('Units','normalized','Position',[0 0 0.8 0.45]);
subplot(1,2,1);
bar(psnr_all'); set(gca,'XTickLabel',case_labels,'XTickLabelRotation',15);
ylabel('PSNR (dB)'); title('PSNR — All Methods');
legend(methods,'Location','northeast','FontSize',7);
grid on; ylim([10 30]);

subplot(1,2,2);
bar(ssim_all'); set(gca,'XTickLabel',case_labels,'XTickLabelRotation',15);
ylabel('SSIM'); title('SSIM — All Methods');
legend(methods,'Location','northeast','FontSize',7);
grid on; ylim([0 0.8]);
sgtitle('Part 3.6 - Final Method Comparison');

figure('Units','normalized','Position',[0 0 1 0.45]);
t = 1;
for m = 1:n_methods
    subplot(1,n_methods,m);
    imshow(im_all{m,t},[0 ref_val]); colormap gray;
    title(sprintf('%s\nPSNR=%.2f\nSSIM=%.4f', ...
        methods{m},psnr_all(m,t),ssim_all(m,t)),'FontSize',7);
end
sgtitle(sprintf('Part 3.6 - All Methods: %s',case_labels{t}));

fprintf('\n========================================\n');
fprintf('       FINAL SUMMARY TABLE\n');
fprintf('========================================\n');
fprintf('%-16s','Method');
for t = 1:3; fprintf('| %-22s',case_labels{t}); end
fprintf('\n%s\n',repmat('-',1,85));
for m = 1:n_methods
    fprintf('%-16s',methods{m});
    for t = 1:3
        fprintf('| PSNR=%.2f  SSIM=%.3f  ',psnr_all(m,t),ssim_all(m,t));
    end
    fprintf('\n');
end
fprintf('========================================\n');