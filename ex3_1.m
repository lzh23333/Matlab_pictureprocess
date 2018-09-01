%空域隐藏与提取
clear all;
load('msg.mat');
load('hall.mat');
[r,c] = size(hall_gray);
hall = double(hall_gray(:));    %将信息载体先转为列向量
l = length(bin_msg);
if(r*c < length(bin_msg))       %防止信息长度大于载体
    l = r*c;
end

a = mod(hall(1:l),2);
hall(1:l) = hall(1:l)-a;        %将前l位最低位全部变为0
hall(1:l) = hall(1:l)+bin_msg(1:l)'; %将信息写入最低位

hall2 = reshape(hall,r,c);      %reshape成原来的图像

%绘图
subplot(1,2,1);
imshow(hall_gray);
title('原图');
subplot(1,2,2);
imshow(uint8(hall2));
title('信息隐藏后的图像');

%信息提取
code = mod(hall2(:),2);                 %取最低位
recover_msg = [];%字符数组
for i = 1:floor(length(code)/8)
    zifu = code(8*i-7:8*i);             %取连续8位
    zifu = zifu.* (2.^(7:-1:0)');       %乘对应的幂
    if(sum(zifu)~=0)
        recover_msg(end+1) = sum(zifu);
    else
        break;                          %说明到达结尾
    end
end
recover_msg = char(recover_msg)         %转为字符串


%进行JPEG编码与解码
jpegcodes = JPEG_encoder(hall2);        
code = JPEG_decoder(jpegcodes);
%套用之前的代码
code = double(mod(code(:),2));
recover_msg = [];%字符数组
for i = 1:floor(length(code)/8)
    zifu = code(8*i-7:8*i);             %取连续8位
    zifu = zifu.* (2.^(7:-1:0)');       %乘对应的幂
    if(sum(zifu)~=0)
        recover_msg(end+1) = sum(zifu);
    else
        break;                          %说明到达结尾
    end
end
recover_msg = char(recover_msg)         %转为字符串