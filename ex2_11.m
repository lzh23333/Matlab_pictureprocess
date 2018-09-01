%ex2-11
clear all;
load('jpegcodes.mat');
pic = picture_recover(binstr2array(jpegcodes.DC_code),...
                      binstr2array(jpegcodes.AC_code),...
                      jpegcodes.H,...
                      jpegcodes.W);     %
subplot(1,2,1);imshow(pic);
title('jpeg');
load('hall.mat');
subplot(1,2,2);
imshow(hall_gray);
title('原图像');
[r,c] = size(hall_gray);
MSE = 1/r/c*sum(sum((hall_gray-pic).^2));
PSNR = 10*log10(255^2/MSE)