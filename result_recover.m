function result = result_recover(DC_code,AC_code,H,W)
%该函数还原出正常的result矩阵，以便后续处理
%参数定义同上
load('JpegCoeff.mat');
DC = DC_decoder(DC_code,DCTAB);         %还原DC码
result = zeros(64,ceil(H/8)*ceil(W/8)); 
result(2:64,:) = AC_decoder(AC_code,ACTAB);%还原AC码
result(1,:) = DC;
end