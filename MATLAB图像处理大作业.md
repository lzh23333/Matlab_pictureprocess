# MATLAB图像处理大作业

#### 						无63	林仲航		2016011051

## 1、基础知识

### 1）、略

### 2）、

	第一题通过rectangle函数在图像上画圆，设置‘Curvature’为1，即对应为圆；

```
load('hall.mat');
temp1 = hall_color;

imshow(temp1);
hold on;
[h,w,d] = size(temp1);              
r = min(h,w)/2;                 %取半径
rectangle('Position',[w/2-r,h/2-r,2*r,2*r],'Curvature',1,'LineWidth',1,'EdgeColor','r');
%画圆
saveas(gca,'temp1.png');
```

	效果如图：

![temp1](D:\github\Matlab_pictureprocess\temp1.png)

	将图片涂成黑白棋盘形状，可以通过将图片分块，根据不同块的坐标决定是否进行涂黑操作。代码与效果如下：

```
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
            temp2(block*i+1:i_end,block*j+1:j_end,1) = 0;          
            temp2(block*i+1:i_end,block*j+1:j_end,2) = 0;
            temp2(block*i+1:i_end,block*j+1:j_end,3) = 0;
            %对三个通道进行涂黑
        end
    end
end
imwrite(temp2,'黑白.png','PNG');%写入文件
```

![黑白](D:\github\Matlab_pictureprocess\黑白.png)



## 2、图像压缩编码

### 1）

	可以在变换域执行。由于对每个像素点减128，相当于减去一个直流分量，也就相当于零频点强度减128*x（x未定）。

	从数学上验证，设A为N*N的待处理矩阵，D为DCT算子，O为全1矩阵，则减去128后得到的C为：
$$
C=D(A-128O)D^T\\
=DAD^T - 128DOD^T\\
=C-{\frac{2}{N}}*128
\left[
\begin{matrix}
\sqrt{\frac{1}{2}} & \sqrt{\frac{1}{2}} & \cdots &\sqrt{\frac{1}{2}}\\
cos\frac{\pi}{2N} & cos\frac{3\pi}{2N}&\cdots &cos\frac{(2N-1)\pi}{2N}\\
\vdots&\vdots&\ddots&\vdots \\
cos \frac{(N-1)\pi}{2N}&cos\frac{(N-1)3\pi}{2N}&\cdots&cos \frac{(N-1)(2N-1)\pi}{2N}
\end{matrix}
\right]

\left[
\begin{matrix}
1 & 1 & \cdots & 1\\
1 & 1 & \cdots & 1\\
\vdots & \vdots & \ddots & \vdots\\
1 & 1 & \cdots & 1
\end{matrix}
\right]
\\
\left[
\begin{matrix}
\sqrt{\frac{1}{2}} & cos\frac{\pi}{2N}& \cdots &cos \frac{(N-1)\pi}{2N}\\
 \sqrt{\frac{1}{2}}& cos\frac{3\pi}{2N}&\cdots &cos\frac{(N-1)3\pi}{2N}\\
\vdots&\vdots&\ddots&\vdots \\
\sqrt{\frac{1}{2}}&cos\frac{(2N-1)\pi}{2N}&\cdots&cos \frac{(N-1)(2N-1)\pi}{2N}
\end{matrix}
\right]
\\
=
C - 128
\left[
\begin{matrix}
\sqrt2 & \sqrt2 &\cdots &\sqrt2\\
0 & 0 & \cdots & 0\\
\vdots & \vdots &\ddots &\vdots\\
0 & 0 & \cdots & 0
\end{matrix}
\right]
\left[
\begin{matrix}
\sqrt{\frac{1}{2}} & cos\frac{\pi}{2N}& \cdots &cos \frac{(N-1)\pi}{2N}\\
 \sqrt{\frac{1}{2}}& cos\frac{3\pi}{2N}&\cdots &cos\frac{(N-1)3\pi}{2N}\\
\vdots&\vdots&\ddots&\vdots \\
\sqrt{\frac{1}{2}}&cos\frac{(2N-1)\pi}{2N}&\cdots&cos \frac{(N-1)(2N-1)\pi}{2N}
\end{matrix}
\right]
\\
=C-128
\left[
\begin{matrix}
N & 0 &  \cdots & 0\\
0 & 0 & \cdots & 0\\
\vdots & \vdots & \ddots & \vdots\\
0 & 0 & \cdots& 0
\end{matrix}
\right]
$$
	从而，即将C~0,0~减去128*N即可；

	代码验证如下：

```matlab
load('hall.mat');               %载入数据
A = double(hall_gray(1:8,1:8)); %转换类型，便于处理

D_1 = 0:7;
D_1 = D_1';
D_2 = 1:2:15;
D = D_1 * D_2;
D = cos(D*pi/2/8);
D(1,:) = D(1,:)/sqrt(2);D
D = D/2;
%生成8*8的DCT算子

C = D*(A-128*ones(8,8))*D'; %所有元素减去128后的DCT变换
C2 = D*A*D';                
C2(1,1) = C2(1,1)-128*8;    %直接在变换域做处理
p = my_equal(C2,C);         %判断是否相等
all(p)                      

```

	结果显示C与C2相等，得证。

![ex2_1_result](D:\github\Matlab_pictureprocess\ex2_1_result.JPG)



### 2）

	由指导书中知识，只需构造DCT算子D，随后对矩阵A进行运算，变换域矩阵C=DAD^T^。若A不为方阵，由构造原理，先对A的每列进行DCT变换，在对变换后的矩阵的每行进行DCT变换，即
$$
C=D_{M*M}A_{M*N}D_{N*N}^T;
$$
	故而在代码中，通过自定义的DCT_operator函数可以方便地生成N维DCT算子，随后进行如上计算即可。

	代码如下：

```matlab
function B = my_dct2(A)
%对A做dct变换
[M,N] = size(A);
D1 = DCT_operator(M);
D2 = DCT_operator(N);
B = D1*A*D2';
```

	DCT_operator函数定义如下：

```matlab
function D = DCT_operator(N)
%返回DCT算子D
[r,c] = size(N);
if(r~=1 && c~=1)
    error('Input must be a single number');
end
D = zeros(N);
D1 = 0:N-1;
D2 = 1:2:2*N-1;
D = D1'*D2;
D = cos(D*pi/2/N);
D(1,:) = D(1,:)/sqrt(2);
D = D*sqrt(2/N);
```

	随意构造一个随机矩阵A，计算my_dct2与dct2的结果，可以发现结果一致；

![ex2_2](D:\github\Matlab_pictureprocess\ex2_2.JPG)



### 3）

	选取hall_gray的120列、120行作为测试图像，做DCT变换后得到系数矩阵C。将C的右4列、左4列分别置为0，随后做逆变换，显示图像结果如下：

![ex2_3](D:\github\Matlab_pictureprocess\ex2_3.png)

	可以看出，将DCT系数矩阵右4列变为0后，图像没有明显变换，但是将左4列变为0，图像明显变黑。从中可以看出人眼对于低频分量的变化较为敏感。且将左4列变为0，相当于去掉了直流分量及低频分量，整体图像变暗。当然，N越大，则右边4列变为0的影响越小。

	代码如下：

```matlab
N=120;
A = double(hall_gray(1:N,1:N))-128;%预处理
B = my_dct2(A);
C = dct2(A);

C2 = C;
C2(:,N-4:N)=0;              %右边4列为0
D = DCT_operator(N);        %N阶DCT算子
A2 = D'*C2*D;               %DCT逆变换
A2 = uint8(A2);             %数据类型变换

subplot(1,3,1);             %绘图
imshow(A2);
title('右4列变0');

subplot(1,3,2);
C3 = C;
C3(:,1:4)=0;                %左4列变0
A3 = D'*C3*D;               %DCT逆变换
imshow(uint8(A3));          
title('左四列变0');

subplot(1,3,3);
imshow(uint8(A));
title('原图');
```



### 4）

	对DCT系数做转置，相当于对原图片进行转置。证明如下：
$$
A'=D^T*C^T*D=D^T*(D*A^T*D^T)*D=A^T
$$
	对DCT矩阵旋转90度，180度，猜测逆变换后图像也有旋转，但不好从数学上说明；

	效果图、代码如下：

```matlab
A = double(hall_gray(1:N,1:N))-128;
D = DCT_operator(N);
C = dct2(A);

%转置
C1 = C';
A1 = D'*C1*D + 128;

%旋转90度
C2 = rot90(C);
A2 = D'*C2*D + 128;

%旋转180度
C3 = rot90(C,2);
A3 = D'*C3*D + 128;

```

![ex2_4](D:\github\Matlab_pictureprocess\ex2_4.png)

	可以看出，旋转后逆变换的图像，除了旋转，相较于原图变化很大，但还是可以大致看出大礼堂的形状。

### 5）

	差分系统表达式可写为：
$$
y(n)=
\begin{cases}
x(n-1)-x(n),&\text{n！=1；}\\
x(1),&\text{n=1;}\\
\end{cases}
$$
	故而
$$
H(z)=\frac{1}{z^{-1}-1}
$$
	代码如下：

```
b = [-1 1];
a = 1;
freqz(b,a,2001);
```

	图像如下：

![freq](D:\github\Matlab_pictureprocess\freq.png)

	由此可见，差分编码系统为高通滤波器。DC系数先进行差分编码，说明高频率分量更多。



### 6）

	观察Category的计算表，可以得知，每个Category的值对应于区间：
$$
[-2^n-1,-2^{n-1}], [2^{n-1},2^n-1]\\
n为category,n>0;
$$

	由此Category的计算公式为：
$$
Category = 
\begin{cases}
floor(log_2|x|)+1, &x!=0;\\
0,&x=0;
\end{cases}
\\x为预测误差
$$


### 7)

	要实现zig_zag，有以下两种思路：

	1.**打表法**，通过写出zigzag扫描得到的列向量对应的8*8矩阵转为的列向量中的下标，即可方便地进行zigzag扫描。然而该方法只适用于固定大小矩阵。

	2.**扫描法**：通过程序直接模拟zigzag扫描，从而可以进行任意大小矩阵的zigzag扫描。具体方法为，定义一个方向指示变量dir，每次循环都会为列向量y添加元素。定义i，j代表矩阵元素下标，每次判断该位置元素是否为边界终止点，从而确定下一次扫描的点的下标所在。	

	综上，定义了zigzag()函数来进行zigzag扫描，扫描方法作为参数传入，详见zigzag.m文件；

```
%模拟zigzag扫描
    dir = 1;        %方向变量，1代表向上扫描，0代表向下扫描
    i = 1;
    j = 1;
    num = 1;
    normal_move = 0;
    for index = 1:r*c
        i,j
        y(index) = A(i,j);
        if(i==1 | j==1 | i==r | j==c) %到达边界
            if( dir & j==c ) %于右边界到达终点
                i = i+1;
                dir = ~dir; %改变方向
            elseif( dir & i==1 )%于上边界到达终点
                j = j+1;
                dir = ~dir;
            elseif( ~dir & i==r )%于下边界达到终点
                j = j+1;
                dir = ~dir;
            elseif( ~dir & j==1 )%于左边界达到终点
                i = i+1;
                dir = ~dir;
                
            else    %说明是起点
                if(dir==1)  %正常移动
                    i=i-1; j=j+1;
                else 
                    i=i+1; j=j-1;
                end
            end
            
        else
            if(dir==1)  %正常移动
                i=i-1; j=j+1;
            else 
                i=i+1; j=j-1;
            end
        end
    end

```

打表方法如下：

```
if(r~=8 | c~= 8)
    error('A must be an 8*8 matrix,otherwise you should use method 1');
end
    
index = [1,9,2,3,10,17,25,18,11,4,5,12,19,26,33,41,...
        34,27,20,13,6,7,14,21,28,35,42,49,57,50,43,36,...      	29,22,15,8,16,23,30,37,44,51,58,59,52,45,38,31,24,32,39,46,53,...
        60,61,54,47,40,48,55,62,63,56,64];
x = A(:);
for i = 1:64
    y(i) = x(index(i));
end
```

	通过验证可知程序正确性：

![zigzag](D:\github\Matlab_pictureprocess\zigzag.JPG)



### 8）

	将测试图像分块，DCT，量化后结果写入矩阵中，每列为一块DCT系数结果zigzag扫描后的列矢量。代码思路大致分为如下步骤：

1. **对图像进行补全，使得行、列数正好为8的倍数；**
2. **利用for循环进行分块，每次选取一个块作为变量block，用于后续处理；**
3. **对block做减去128的预处理；**
4. **对block进行DCT变换；**
5. **对DCT系数进行zigzag扫描后，结果写入对应的结果矩阵中。**	



	具体代码如下（详见ex2_8.m)，结果存于ex2_8_result中：

```matlab
A = double(hall_gray);

%先对测试图像进行补全
[r c] = size(A);
if(mod(r,8)~=0)
    A(r+1:(floor(r/8)+1)*8,:)=0;
end
if(mod(c,8)~=0)
    A(:,c+1:(floor(c/8)+1)*8)=0;
end

%随后开始分块
[r c] = size(A);
result = zeros(64,r*c/64);
for i = 1:r/8
    for j = 1:c/8
        block = A(8*(i-1)+1:8*i,8*(j-1)+1:8*j);
        %分块完成
        block = block-128;  %预处理
        D = dct2(block);    %DCT变换
        D = round(D./QTAB); %量化
        result(:,(i-1)*c/8+j)=zigzag(D,2);    %zigzag扫描
    end
end

```

