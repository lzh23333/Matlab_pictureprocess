function y = binstr2array(binstream)
%将二进制码流（字符串）转为数组
if(~ischar(binstream))
    error('Input must be a binstream char!');
end

y = zeros(length(binstream),1);
for i = 1:length(binstream)
    if(binstream(i)=='1')
        y(i) = 1;
    elseif(binstream(i)~='0')
        error('二进制码流存在非二进制符号!');
    end
end
