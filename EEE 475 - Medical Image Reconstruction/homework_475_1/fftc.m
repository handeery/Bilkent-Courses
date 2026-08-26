function d = fftc(im)
% d = fftc(im)
%
% fftc performs a centered 1D FFT

d = fftshift(fft(ifftshift(im)));

end