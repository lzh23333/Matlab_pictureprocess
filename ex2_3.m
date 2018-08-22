%ex2_3
clear all;
load('hall.mat');
N=120;
A = double(hall_gray(1:N,1:N))-128;
B = my_dct2(A);
C = dct2(A);


C2 = C;
C2(:,N-4:N)=0;              %右边4列为0
D = DCT_operator(N);        %N阶DCT算子
A2 = D'*C2*D;               %DCT逆变换
A2 = uint8(A2);             %数据类型变换

subplot(1,3,1);             %绘图
imshow(A2);
title('右4列变0');

subplot(1,3,2);
C3 = C;
C3(:,1:4)=0;                %左4列变0
A3 = D'*C3*D;               %DCT逆变换
imshow(uint8(A3));          
title('左四列变0');

subplot(1,3,3);
imshow(uint8(A));
title('原图');
p = my_equal(B,C);
