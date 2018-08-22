%ex2_3
clear all;
load('hall.mat');
N=120;
A = double(hall_gray(1:N,1:N))-128;
D = DCT_operator(N);
C = dct2(A);

%转置
C1 = C';
A1 = D'*C1*D + 128;

%旋转90度
C2 = rot90(C);
A2 = D'*C2*D + 128;

%旋转180度
C3 = rot90(C,2);
A3 = D'*C3*D + 128;

%显示图片
subplot(2,2,1);
imshow(uint8(A+128));
title('原图');
subplot(2,2,2);
imshow(uint8(A1));
title('转置后');
subplot(2,2,3);
imshow(uint8(A2));
title('旋转90度');
subplot(2,2,4);
imshow(uint8(A3));
title('旋转180度');
