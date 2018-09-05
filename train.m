%train.m
%对Faces中的图片进行循环，从而得到标准特征v
%v为4*1cell数组，每个cell对应一个L下的特征v
num = 33;
L = 0;
v = cell(4,1);
for L = 3:6
    v{L-2} = zeros(2^(3*L),1);
    for i = 1:num
        v{L-2} = v{L-2} + feature_extract('Faces/',strcat(num2str(i),'.bmp'),L);
    end
v{L-2} = v{L-2}/num;
end
%得到v
%{
plot(v);
title('特征--概率密度');
xlabel('index');
ylabel('probability');
%}
save face_standard.mat v