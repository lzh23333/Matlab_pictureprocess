function B = my_dct2(A)
%对A做dct变换
[M,N] = size(A);
D1 = DCT_operator(M);
D2 = DCT_operator(N);
B = D1*A*D2';