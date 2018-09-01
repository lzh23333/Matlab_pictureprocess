function y = AC_decoder(AC_code,ACTAB)
%y为翻译好的AC系数矩阵
%AC_code为AC码流
%ACTAB为翻译表

y = [];
ACTAB = [0,0,4,1,0,1,0,zeros(1,12);ACTAB;15,0,11,1,1,1,1,1,1,1,1,0,0,1,0,0,0,0,0];  %加入一个EOB与ZRL
[r,c] = size(ACTAB);
AC_table = [2*ones(r,1),ACTAB];         
tree = build_huffmantree(AC_table);     %构建Huffman树
save tree 

index = 1;                          %AC码的index
tree_index = 1;                     %tree节点序号
find_end = 0;                       %找到了一段Huffman码
c_index = 1;                        %代表result矩阵的第几列
r_index = 1;                        %代表result矩阵的第几行

while(index <= length(AC_code))
    if(AC_code(index)==0)
        tree_index = tree(tree_index).left;
        if(length(tree(tree_index).value)>1)
            find_end = 1;       %该段解码完成
        end
    elseif(AC_code(index)==1)
        tree_index = tree(tree_index).right;
        if(length(tree(tree_index).value)>1)
            find_end = 1;       %该段解码完成
        end
    else
        error('error!');
    end
    index = index + 1;
    
    if(find_end)
        %tree(tree_index)
       
        run = tree(tree_index).value(1) ;       %确认游程数
        AC_size = tree(tree_index).value(2);    %确认size
        amplitude = 0;
        tree_index = 1;                         %回到树的root
        find_end = 0;
        number = 0;
        
        if(run==0 & AC_size==0)
            %为一个EOB
            
           
            c_index = c_index + 1;
            y(r_index:63,c_index-1) = 0;
            r_index = 1;
            %index
        else
            number = 0;
            if(AC_size ~= 0)
                number = AC_code(index:index+AC_size-1);
                index = index + AC_size;
            else
                %说明为ZRL
                if(run==15)
                   index = index;   %保持不变
                else
                    error('run should be 15!');
                end
                
            end
            %更新index
            
           
            is_neg = 0;
            if(number(1)==0 & AC_size~=0)
                is_neg = 1;
                number = double(~number);
            end
            for i = 1:length(number)
                number(i) = number(i)*(2^(length(number)-i));
                %各位乘对应的系数
            end
            
            amplitude = sum(number);
            if(is_neg)  amplitude = -amplitude; end
      
            y(r_index:r_index+run-1,c_index) = 0;%填入连续的0
            y(r_index+run,c_index)=amplitude;   %填入amplitude
            r_index = r_index+run+1;            %更新r_index
        
        end
    end
end

end

        
    
    
    
    
    
    
    
    
    