function y = my_equal(A,B)
%判断两个数是否相等
%只要误差小于阈值即可
if(size(A) ~= size(B))
    error('A and B are not the same size');
end
[r,c] = size(A);
y = zeros(size(A));
for i = 1:r
    for j = 1:c
        if(abs(A(i,j)-B(i,j)) < 1e-4)
            y(i,j) = 1;
        else
            y(i,j) = 0;
        end
    end
end