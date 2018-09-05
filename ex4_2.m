%ex4_2

L = 4;
thresholds = [0.4,0.24,0.17,0.1];
for L = 3:6
    subplot(2,2,L-2);
    img = imread('test2.jpg');
    [y,num,identify] = face_detect(img,L,thresholds(L-2));
    title(['L=',num2str(L),',threshold=',num2str(thresholds(L-2))]);    
        
   
    %{
    load('face_standard.mat');
    v = v{L-2};
    thres = 0;
    time = 0;
    for i = 1:33
        img = imread(strcat('Faces/',num2str(i),'.bmp'));
        imshow(img);
        u = feature_extract(img,L);
        sqrt(u'*v);
        thres = thres + sqrt(u'*v);
        time = time + 1;
    end
    thres = thres/time
    %}
end


%img = imread('test2.jpg');
%[y,num,identify] = face_detect(img,L,threshold);