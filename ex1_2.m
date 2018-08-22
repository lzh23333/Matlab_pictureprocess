%1.2练习题
load('hall.mat');
temp1 = hall_color;
temp2 = hall_color;

imshow(temp1);
hold on;
[h,w,d] = size(temp1);              
r = min(h,w)/2;                 %取半径
rectangle('Position',[w/2-r,h/2-r,2*r,2*r],'Curvature',1,'LineWidth',1,'EdgeColor','r');
%画圆
saveas(gca,'temp1.png');


block = 10;                 %单个格子的长度
for i = 0:floor(h/block)
    for j = 0:floor(w/block)
        if(mod(i+j,2)==1)
            i_end = (i+1)*block;
            j_end = (j+1)*block;
            if(h<(i+1)*block)
                i_end = h;
            end
            if(w<(j+1)*block)
                j_end = w;
            end
            
            temp2(block*i+1:i_end,block*j+1:j_end,1) = 0;           %涂黑
            temp2(block*i+1:i_end,block*j+1:j_end,2) = 0;
            temp2(block*i+1:i_end,block*j+1:j_end,3) = 0;
        end
    end
end
imwrite(temp2,'黑白.png','PNG');