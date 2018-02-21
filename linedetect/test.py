import cv2
import numpy as np



I = cv2.imread('ss.jpg')
I_gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
I_gray = cv2.equalizeHist(I_gray)
I = np.float32(I_gray)

# I = np.log(I +1.0)

size = I.shape
print(size)

ay = np.arange(0,size[0])
ax = np.arange(0, size[1])

cY = np.ceil(size[0]/2)
cX = np.ceil(size[1]/2)

xx, yy = np.meshgrid(ax, ay)

GausN = (xx - cX)**2 + (yy -  cY)**2
sigma = 10.0
sigma2 = 60.0

H = np.exp(-GausN/(2*sigma**2))
H2 = np.exp(-GausN/(2*sigma2**2))

H = 1 - H
H = np.fft.fftshift(H)

# H = H / np.sum(H)


# Hs = np.fft.fftshift(H)
If = cv2.dft(I,flags = cv2.DFT_COMPLEX_OUTPUT)
# If = np.fft.fft(I)
Ifs = np.fft.fftshift(If)
Ifs = If
Rfs = np.ones(Ifs.shape)
Rfs[:,:,0] = Ifs[:,:,0]* H
Rfs[:,:,1] = Ifs[:,:,1]* H

Rfs_img = Rfs[:,:,0]
Rfs_img = Rfs_img/np.max(Rfs_img)*255
Ifs_img = Ifs[:,:,0]
Ifs_img = Ifs_img/np.max(Ifs_img)*255
Rf = Rfs
# Rf = np.fft.ifftshift(Rfs)
R = cv2.idft(Rf,flags = cv2.DFT_SCALE)
# R = np.fft.ifft(Rf) 

Rr = cv2.magnitude(R[:,:,0],R[:,:,1])
# Rr = np.abs(R)
# print(np.max(Rr))
# Rr = Rr/np.max(Rr)*255
# Rr = np.exp(Rr) - 1
# print(np.max(Rr))
Rr = Rr/np.max(Rr)*255
Ri = np.uint8(Rr)
Res = Ri
# RiEq = cv2.equalizeHist(Ri)

cv2.imshow('22',Res)
cv2.imshow('23',H)
cv2.imshow('Rfs_img',Rfs_img)
cv2.imshow('Ifs_img',Ifs_img)


cv2.waitKey()