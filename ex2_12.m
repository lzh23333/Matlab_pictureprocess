%ex2_12
load('snow.mat');
jpegcodes = JPEG_encoder(snow); %压缩

pic = picture_recover(binstr2array(jpegcodes.DC_code),...
                      binstr2array(jpegcodes.AC_code),...
                      jpegcodes.H,...
                      jpegcodes.W);     %解压缩
[r,c] = size(snow);
MSE = 1/r/c*sum(sum((snow-pic).^2));
PSNR = 10*log10(255^2/MSE)              %计算图像质量

subplot(1,2,1);
imshow(pic);
title('jpeg');
subplot(1,2,2);
imshow(snow);
title('原图像');