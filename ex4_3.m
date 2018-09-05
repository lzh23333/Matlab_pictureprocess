%ex4-3

L = 4;
threshold = 0.27;

subplot(2,2,1);
img = imread('test.jpg');
[y,num,identify] = face_detect(img,L,thresholds(L-2));
title('原图');

subplot(2,2,2);
img2 = rot90(img);
[y,num,identify] = face_detect(img2,L,thresholds(L-2));
title('旋转90度');

subplot(2,2,3);
img3 = imresize(img,[size(img,1) 2*size(img,2)],'nearest');
[y,num,identify] = face_detect(img3,L,thresholds(L-2));
title('调整图像宽度');

subplot(2,2,4);
img4 = imadjust(img,[.2 .3 0; .6 .7 1],[]);
[y,num,identify] = face_detect(img4,L,thresholds(L-2));
title('调整图像颜色');