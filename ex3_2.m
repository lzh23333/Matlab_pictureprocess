%ex3_2
clear all;
load('hall.mat');
pic = double(hall_gray);
msg = 'Tsinghua University';

subplot(2,2,1);
imshow(hall_gray);
title('Original Picture');


for method = 1:3
    method
    jpegcodes = msg_hide(pic,msg,method);
    subplot(2,2,1+method);
    
    y = JPEG_decoder(jpegcodes);
    imshow(y);
    title(strcat('方法:',method+'0'));
    msg2 = msg_take(jpegcodes,method);
    disp('message hidden in the picture is:');
    disp(msg2);
    
    
end

    