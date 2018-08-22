function D = DCT_operator(N)
%返回DCT算子D
[r,c] = size(N);
if(r~=1 && c~=1)
    error('Input must be a single number');
end
D = zeros(N);
D1 = 0:N-1;
D2 = 1:2:2*N-1;
D = D1'*D2;
D = cos(D*pi/2/N);
D(1,:) = D(1,:)/sqrt(2);
D = D*sqrt(2/N);