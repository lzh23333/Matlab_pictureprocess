function y = zigzag(A,method)
%该函数为zigzag扫描函数
%返回值为经zigzag扫描后得到的列向量
%method为扫描采用算法，1为模拟，2为打表
%A为矩阵
if(~ismatrix(A))
    error('input must be a matrix!');
end
[r,c] = size(A);
y = zeros(r*c,1);

skip = 0;
if(r==1 | c==1)
    y = A(:);
    skip = 1;
end
%若已经是列向量或者行向量

if(method == 1 & ~skip)
    %模拟zigzag扫描
    dir = 1;        %方向变量，1代表向上扫描，0代表向下扫描
    i = 1;
    j = 1;
    num = 1;
    normal_move = 0;
    for index = 1:r*c
        y(index) = A(i,j);
        if(i==1 | j==1 | i==r | j==c) %到达边界
            if( dir & j==c ) %于右边界到达终点
                i = i+1;
                dir = ~dir; %改变方向
            elseif( dir & i==1 )%于上边界到达终点
                j = j+1;
                dir = ~dir;
            elseif( ~dir & i==r )%于下边界达到终点
                j = j+1;
                dir = ~dir;
            elseif( ~dir & j==1 )%于左边界达到终点
                i = i+1;
                dir = ~dir;
                
            else    %说明是起点
                if(dir==1)  %正常移动
                    i=i-1; j=j+1;
                else 
                    i=i+1; j=j-1;
                end
            end
            
        else
            if(dir==1)  %正常移动
                i=i-1; j=j+1;
            else 
                i=i+1; j=j-1;
            end
        end
    end
elseif(method==2)
    if(r~=8 | c~= 8)
        error('A must be an 8*8 matrix,otherwise you should use method 1');
    end
    
    index = [1,9,2,3,10,17,25,18,11,4,5,12,19,26,33,41,...
        34,27,20,13,6,7,14,21,28,35,42,49,57,50,43,36,...
        29,22,15,8,16,23,30,37,44,51,58,59,52,45,38,31,24,32,39,46,53,...
        60,61,54,47,40,48,55,62,63,56,64];
    x = A(:);
    for i = 1:64
        y(i) = x(index(i));
    end
end
                
            
                
                
            