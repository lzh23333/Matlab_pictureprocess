function pic = picture_recover(DC_code,AC_code,H,W)
%输入压缩完的JPEG
%DC_code、AC_code存于一维向量中
%H为图像的高度
%W为图像的宽度
%输出pic为灰度图

load('JpegCoeff.mat');
result2 = result_recover(DC_code,AC_code,H,W);
save('ex2_11_result.mat','result2');

r = ceil(H/8);
c = ceil(W/8);

pic = zeros(r*8,c*8);   %初始化图像块

for i = 1:size(result2,2)
    column = result2(:,i);          %取一列
    block = anti_zigzag(column);%还原图像块
    block = block.*QTAB;        %反量化
    %block = block.*QTAB/2;      %反量化，QTAB为原来一半
    pic_block = idct2(block);   %逆DCT2
    
    r_index = ceil(i/c);
    c_index = mod(i,c);
    if(c_index == 0)
        c_index = c;
    end
    %确定图像块所处的位置
    pic(8*r_index-7:8*r_index,8*c_index-7:8*c_index) = pic_block;
    %拼接
end
pic = uint8(pic(1:H,1:W)+128);
end



