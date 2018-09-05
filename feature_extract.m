function v = feature_extract(varargin)
%v = feature_extract(path,pic_name)
%path为路径
%pic_name为文件名
%v为对该图提取的特征向量
%L为颜色位数的个数，1<=L<=8,但是推荐L<=6

%另一种输入如下：
%v = feature_extract(pic,L)
%pic为rgb矩阵
%L同上

pic = [];
L = 0;

if nargin == 3
    path = varargin{1};
    pic_name = varargin{2};
    L = varargin{3};
    if(exist(strcat(path,pic_name))==0)
        error('No such file or you should check path/pic_name');
    end
    %判定文件是否存在

    pic = imread(strcat(path,pic_name));

    if(size(pic,3)~=3)
        error('picture should be a rgb form');
    end
    %判定是否为rgb图片

    if(L>6 )
        error('L is recommander to be smaller than 7');
    end
    %L检查

   
    
elseif nargin==2
    pic = varargin{1};
    L = varargin{2};
end

pic = double(reshape(pic,size(pic,1)*size(pic,2),1,3));
v = zeros(2^(3*L),1);
basic = 2^(8-L);
%初始化
 
for i = 1:size(pic,1)
    a = [pic(i,1,1),pic(i,1,2),pic(i,1,3)]; %取出像素
    index = sum(floor(a/basic).*(2.^(2*L:-L:0)))+1; %计算该颜色对应的下标
    v(index) = v(index) + 1;            %次数+1
end
v = v/size(pic,1);                      
%循环统计每个颜色出现的次数并求出频率
