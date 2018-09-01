function msg = msg_take(pic_jpeg,method)
%从图片jpeg码中读取信息
%method定义同msg_hide
%返回值msg为字符

%------------------获取DCT系数矩阵-----------------------------
result = result_recover(binstr2array(pic_jpeg.DC_code),...
                        binstr2array(pic_jpeg.AC_code),...
                        pic_jpeg.H,pic_jpeg.W);


%------------------信息提取-----------------------------------
switch method
    case 1
        result = result(:);         %转为列向量
      
        msg_array = mod(result,2);  
        
        for i = 1:floor(length(msg_array)/8)
            msg(i) = sum(msg_array((i-1)*8+1:i*8).*(2.^(7:-1:0)'));  %计算出第i为的ascii码
        end
    
        msg = char(msg);
    case 2
        result = result(:);         %转为列向量
        msg_array = mod(result,2);  
        for i = 1:floor(length(msg_array)/8)
            msg(i) = sum(msg_array((i-1)*8+1:i*8).*(2.^(7:-1:0)'));  %计算出第i为的ascii码
            if(msg(i)==0)           %到达结束位
                break;
            end
        end
        msg = char(msg);
    case 3
        msg_array = [];
        for i = 1:size(result,2)    
            column = result(:,i);   %取result矩阵一列
            find = 0;
            index = size(result,1);
            while(~find)
                if(column(index)~=0)
                    find=1;
                else
                    index = index-1;
                end
            end
            %找到最后一位
            if(column(index)==-1)
                msg_array(i) = 0;
            elseif(column(index)==1)
                msg_array(i) = 1;
            else
                break
            end
        end
        for i = 1:floor(length(msg_array)/8)
            msg(i) = sum(msg_array((i-1)*8+1:i*8).*(2.^(7:-1:0)));  %计算出第i为的ascii码
            if(msg(i)==0)
                break;
            end
        end
    otherwise
        error('no such method!');
end
msg = char(msg);
end
