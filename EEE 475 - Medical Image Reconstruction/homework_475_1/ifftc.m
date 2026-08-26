function im = ifftc(d)
% im = ifftc(d)
%
% ifftc performs a centered 1D inverse FFT

im = fftshift(ifft(ifftshift(d)));

end