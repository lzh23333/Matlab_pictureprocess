function DC_code = DC_coeff(DC_vector)
%输入直流分量的向量DC_vector
%输出DC为其对应的二进制码流

D = zeros(length(DC_vector),1);
D(2:length(DC_vector)) = -diff(DC_vector);
D(1) = DC_vector(1);
%差分编码

DC_code = '';
load('JpegCoeff.mat');
for i = 1:length(D)
    DC_code = strcat(DC_code,DC_translate(D(i),DCTAB));
    
end
end

function y = DC_translate(c,DCTAB)
%y为DC系数翻译的二进制字符串
%c为预测误差
%DCTAB即为对应的码表
    y = '';
    cor = 0;    %cor为Category
    if(c~=0)
        cor = floor(log2(abs(c)))+1;
    end
    s_length = DCTAB(cor+1,1);
    
    for i = 1:s_length
        y = strcat(y,DCTAB(cor+1,1+i)+'0');
    end
    %Huffman编码
    
    s = dec2bin(abs(c));
    if(c<0)
        for i = 1:length(s)
            if(s(i)=='1')
                s(i)='0';
            elseif(s(i)=='0')
                s(i)='1';
            end
        end
    end
    %预测误差的二进制码   
    
    y = strcat(y,s);
    
    
end