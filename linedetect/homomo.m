I = imread('ss.jpg');
% imshow(I)
I = im2double(I);
% I = log(1 + I);
% imshow(I)
M = size(I,1);
N = size(I,2);
sigma = 3;
sigma2 = 100;
[X, Y] = meshgrid(1:N,1:M);
centerX = ceil(N/2);
centerY = ceil(M/2);
gaussianNumerator = (X - centerX).^2 + (Y - centerY).^2;
H = exp(-gaussianNumerator./(2*sigma.^2));
H2 = exp(-gaussianNumerator./(2*sigma2.^2));
H = (1 - H);
figure;
imshow(H)
% H = fftshift(H);
If = fft2(I, M, N);
Ifs = fftshift(If);
Rfs = H.*Ifs
R = ifftshift(Rfs)
Rf=ifft2(R)
Iout = real(Rf);
Iout = Iout(1:size(I,1),1:size(I,2));
% Ihmf = exp(Iout) - 1;
figure;
imshowpair(I, Iout, 'montage')