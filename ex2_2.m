%ex2_2
clear all;
load('hall.mat');
A = double(hall_gray(1:8,1:8));
B = my_dct2(A);
C = dct2(A);
p = my_equal(B,C);