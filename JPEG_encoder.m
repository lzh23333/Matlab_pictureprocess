function jpegcodes = JPEG_encoder(picture)
%对picture(灰度图)进行jpeg编码
%y为结构体，分别由DC_code,AC_code,H,W,压缩比ratio组成
load('JpegCoeff.mat');
A = double(picture);

%先对测试图像进行补全
[r c] = size(A);
H = r;                      %高
W = c;                      %图像宽度
if(mod(r,8)~=0)
    A(r+1:(floor(r/8)+1)*8,:)=0;
end
if(mod(c,8)~=0)
    A(:,c+1:(floor(c/8)+1)*8)=0;
end

%随后开始分块
[r c] = size(A);
result = zeros(64,r*c/64);
for i = 1:r/8
    for j = 1:c/8
        block = A(8*(i-1)+1:8*i,8*(j-1)+1:8*j);
        %分块完成
        block = block-128;  %预处理
        D = dct2(block);    %DCT变换
        D = round(D./QTAB); %量化
        %D = round(D./QTAB*2);   %QTAB减半后量化
        result(:,(i-1)*c/8+j)=zigzag(D,2);    %zigzag扫描
    end
end

DC = result(1,:);           %第一行即为DC系数

DC_code = DC_coeff(DC);     %DC码
AC_code = '';               %AC码
for i = 1:r*c/64            %逐块翻译AC码
    AC_code = strcat(AC_code,AC_coeff(result(2:end,i)));
end

jpegcodes = struct('DC_code',{DC_code},'AC_code',AC_code,'H',H,'W',W);

%以下计算压缩比
[r,c] = size(picture);
pic_size = r*c;             %计算原始图像字节数
code_length = length(jpegcodes.DC_code)+length(jpegcodes.AC_code);%计算码流长度
ratio = pic_size * 8 / code_length ;   %字节数乘8后除于码流长度即为压缩比
jpegcodes.ratio = ratio;        