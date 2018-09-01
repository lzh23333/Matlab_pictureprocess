%generate message
message = 'Tsinghua University';
bin_msg = [];
for i = 1:length(message)
    msg = binstr2array(dec2bin(abs(message(i))))';  %转为二进制数组
    msg = [zeros(1,8-length(msg)),msg];             %每个字符对应8位
    bin_msg(end+1:end+8) = msg;                     
end
bin_msg(end+1:end+8)=zeros(1,8);
save msg.mat bin_msg