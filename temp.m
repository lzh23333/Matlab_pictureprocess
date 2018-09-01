%for verify
clear all;
load('JpegCoeff.mat');
num = 0;

load('hall.mat');
pic = double(hall_gray);
[H,W]=size(pic);
result = DCT_result(pic);
[r,c] = size(result);
msg_array = [1,0,1,0,1,0,1,0];
result = result(:);         %result变为列向量
result = result - mod(result,2);
%使每个DCT系数的最后一位都为0
if(length(msg_array) < length(result))
    msg_array(end+1:end+length(result)-length(msg_array))=0; %使msg_array与result等长
    end
result = result + msg_array(1:length(result))';   %修改每一位DCT系数的最低位
result = reshape(result,r,c);
DC = result(1,:);           %第一行即为DC系数
DC_code = DC_coeff(DC);     %DC码
AC_code = '';               %AC码
for i = 1:c            %逐块翻译AC码
    AC_code = strcat(AC_code,AC_coeff(result(2:end,i)));
end
jpegcodes = struct('DC_code',{DC_code},'AC_code',AC_code,'H',H,'W',W);

load('JpegCoeff.mat');
DC = DC_decoder(binstr2array(DC_code),DCTAB);         %还原DC码
x = zeros(64,ceil(H/8)*ceil(W/8)); 
x(2:64,:) = AC_decoder(binstr2array(AC_code),ACTAB);%还原AC码
x(1,:) = DC;

isequal(x,result)

y = JPEG_decoder(jpegcodes);      %解码，得到待隐藏信息图片
result2 = DCT_result(y);
isequal(result2,result)
a = result2(:);
a(1:48)'
b = result(:);
b(1:48)'