function y = AC_coeff(AC_vector)
%输入AC_vector为经过量化的待处理的AC系数
%输出y为对应的二进制码流
%只对第一列进行处理
[r,c]=size(AC_vector);
if(c~=1)
    sprintf('AC_vector shoule be a column vector, otherwise you will only get one column result;')
end

load('JpegCoeff.mat');
run = 0;
y = '';
ZRL = '11111111001';    %16连0
EOB = '1010';           %块结束符

for i = 1:length(AC_vector)
    if(AC_vector(i)==0)
       run = run+1;
    else
        if(run < 16)
            y = strcat(y,AC_translate(run,AC_vector(i),ACTAB));  %添加该非0系数的二进制码
            run = 0;
        else
            while(run>=16)
                y = strcat(y,ZRL);
                run = run-16;
            end
            y = strcat(y,AC_translate(run,AC_vector(i),ACTAB)); 
            run = 0;                    %清零
        end
    end
end
y = strcat(y,EOB);  %在结尾增加EOB

end

function y = AC_translate(run,c,ACTAB)
%该函数为子函数
%run为游程数
%c为非零AC系数
%ACTAB为Huffman对照表
%返回值y对应AC系数二进制码

size = 0;
if(c ~= 0)
    size = floor(log2(abs(c)))+1;
end
%确定该系数的size

amplitude = dec2bin(abs(c));
if(c<0)
    for i = 1:length(amplitude)
        if(amplitude(i)=='0')
            amplitude(i)='1';
        elseif(amplitude(i)=='1')
            amplitude(i)='0';
        end
    end
end
%确定amplitude

huffman = '';
row = run*10 + size;
l = ACTAB(row,3);
for i = 1:l
    huffman = strcat(huffman,ACTAB(row,3+i)+'0');
end
%确定run/size的huffman编码

y = strcat(huffman,amplitude);
%返回值
end

