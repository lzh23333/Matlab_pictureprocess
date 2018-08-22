clear all;
load('hall.mat');               %载入数据
A = double(hall_gray(1:8,1:8)); %转换类型，便于处理

D_1 = 0:7;
D_1 = D_1';
D_2 = 1:2:15;
D = D_1 * D_2;
D = cos(D*pi/2/8);
D(1,:) = D(1,:)/sqrt(2);D
D = D/2;
%生成8*8的DCT算子

C = D*(A-128*ones(8,8))*D'; %所有元素减去128后的DCT变换
C2 = D*A*D';                
C2(1,1) = C2(1,1)-128*8;    %直接在变换域做处理
p = my_equal(C2,C);         %判断是否相等
all(p)                   
C2
C