%ex2_8
%对测试图像进行分块，DCT以及量化
clear all;
load('JpegCoeff.mat');
load('hall.mat');

A = double(hall_gray);

%先对测试图像进行补全
[r c] = size(A);
if(mod(r,8)~=0)
    A(r+1:(floor(r/8)+1)*8,:)=0;
end
if(mod(c,8)~=0)
    A(:,c+1:(floor(c/8)+1)*8)=0;
end

%随后开始分块
[r c] = size(A);
result = zeros(64,r*c/64);
for i = 1:r/8
    for j = 1:c/8
        block = A(8*(i-1)+1:8*i,8*(j-1)+1:8*j);
        %分块完成
        block = block-128;  %预处理
        D = dct2(block);    %DCT变换
        D = round(D./QTAB); %量化
        result(:,(i-1)*c/8+j)=zigzag(D,2);    %zigzag扫描
    end
end

