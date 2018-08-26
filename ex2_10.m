%计算压缩比
clear all;
load('hall.mat');
load('jpegcodes');

[r,c] = size(hall_gray);
pic_size = r*c;             %计算原始图像字节数
code_length = length(jpegcodes.DC_code)+length(jpegcodes.AC_code);%计算码流长度
ratio = pic_size * 8 / code_length    %字节数乘8后除于码流长度即为压缩比
code_length2 = code_length + 16;