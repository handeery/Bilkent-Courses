function im = ifft2c(d)
% im = ifft2c(d)
%
% ifft2c performs a centered 2D inverse FFT

im = fftshift(ifft2(ifftshift(d)));

end