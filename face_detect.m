function [y, num,identify] = face_detect(img,L,threshold)
% [y, num] = face_detect(img)
% img is an rgb image
% y is an rgb image with num people's faces detected
% L is the bits of color

if(size(img,3)~=3)
    error('img must be an rgb image!');
end

%------------initial part-----------%
load('face_standard.mat');
if(L>6)L=6;end
if(L<3)L=3;end
v = v{L-2};
num = 0;
size_img = size(img,1);
if(size_img > size(img,2))
    size_img = size(img,2);
end

block_l = floor(size_img/16);
if(block_l > 50) block_l = 50;
elseif(block_l < 16)block_l = 16;
end
%limit on block_l 
%plot(v);

%----------- divide img and process------------%
[r,c,h] = size(img);
H = ceil(r/block_l); W = ceil(c/block_l);
identify = zeros(H,W);  %(i,j)=1 means that block is face

y = img;
imshow(y);
for i = 1:H
    for j = 1:W
        right = block_l*j;
        down = block_l*i;
        left = block_l*(j-1)+1;
        up = block_l*(i-1)+1;
        if(right > c)right = c;end
        if(down > r) down = r;end
        
        block = img(up:down,left:right,:);  %dividing part finished
        u = feature_extract(block,L);
        %{
        plot(u,'r');
        hold on;
        plot(v,'g');
        hold off;
        pause
        %}
        cor = sqrt(u)'*sqrt(v);
        if cor >= threshold
            identify(i,j) = 1;
            rectangle('Position',[left,up,right-left,down-up],'LineWidth',1,'EdgeColor','r');
        end 
    end
end

%-------------------block faces-----------------------
%{
waitlist_r = [];
waitlist_c = [];
num = 1;
while(size(find(identify==1),1) > 0)
    %1代表未分类
    %0代表
   
    if(length(waitlist_r)==0)
        [I,J] = find(identify==1);
        index = randi(length(I));
        waitlist_r(end+1) = I(index);
        waitlist_c(end+1) = J(index);
        num = num+1;
    else
        r = waitlist_r(1);waitlist_r = waitlist_r(2:end);
        c = waitlist_c(1);waitlist_c = waitlist_c(2:end); 
        identify(r,c)=num;
    
        if(r+1 < size(identify,1) & identify(r+1,c)==1)
            waitlist_r(end+1) = r+1;
            waitlist_c(end+1) = c;
            identify(r+1,c) = num;
        end
        if(c+1 < size(identify,2) & identify(r,c+1)==1)
            waitlist_r(end+1) = r;
            waitlist_c(end+1) = c+1;
            identify(r,c+1) = num;
        end
        if(r-1 > 0 & identify(r-1,c)==1)
            waitlist_r(end+1) = r-1;
            waitlist_c(end+1) = c;
            identify(r-1,c) = num;
        end
        if(c-1 > 0 & identify(r,c-1)==1)
            waitlist_r(end+1) = r;
            waitlist_c(end+1) = c-1;
            identify(r,c-1) = num;
        end
    end
end
%以上，将不同类的人脸方块分开，从而进行接下来的操作
ignore = 2:num;
sum_2 = zeros(size(ignore));
for i = 2:num
    sum_2(i-1) = length(find(identify==i));
end

tuan = max(sum_2);
for i = 1:num-1
    if(sum_2(i)/tuan < 0.05)
        ignore(i) = 1;
    else
        ignore(i) = 0;
    end
end

for i = 2:num
    if(ignore(i-1)==0)
       [I,J] = find(identify==i);
       up = (min(I)-1)*block_l + 1;
       down = max(I)*block_l;
       left = (min(J)-1)*block_l + 1;
       right = max(J)*block_l;
       rectangle('Position',[left,up,right-left,down-up],'LineWidth',1,'EdgeColor','r');
    end
end

%}





