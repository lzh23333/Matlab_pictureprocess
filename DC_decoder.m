function y = DC_decoder(DC_code,DCTAB)
%y为翻译好的预测误差列向量
%DCTAB为翻译表
%DC_code为DC码流
y = [];
[r,c] = size(DCTAB);
DC_table = [ones(r,1),[0:r-1]',DCTAB];  %得到符合自定义函数格式的DC_table
tree = build_huffmantree(DC_table);     %得到huffman树

index_set = [];


index = 1;          
tree_index = 1;     %用于指示树中节点位置
find_end = 0;       %用于指示是否完成一段的解码
while(index < length(DC_code))
    if(DC_code(index)==0)
        tree_index = tree(tree_index).left;
        if(tree(tree_index).value~=-1)
            find_end = 1;       %该段解码完成
        end
    elseif(DC_code(index)==1)
        tree_index = tree(tree_index).right;
        if(tree(tree_index).value~=-1)
            find_end = 1;
        end
    else
        error('DC_code error!');
    end
    index = index + 1;
    %找到结尾的处理
    if(find_end)
        index_set(end+1) = index;
        category = tree(tree_index).value;
        tree_index = 1;             %重回根节点
        find_end = 0;               %更新
        number = 0;                 %number为预测误差二进制码
        if(category~=0)
            number = DC_code(index:index+category-1);
            index = index + category;
        else
            index = index + 1;      %更新index
        end
        
        pre_error = 0;              %预测误差
        is_neg = 0;                 %是否为负数
   
        if(number(1)==0 & category~=0)  %说明该预测误差为负数
            number = double(~number);   %按位取反
            is_neg = 1;
        end
        
        for i = 1:length(number)
            number(i) = number(i)*(2^(length(number)-i));
            %各位乘对应的系数
        end

        pre_error = sum(number);
        if(is_neg)
            pre_error = -pre_error;
        end
        %得到预测误差
        y(length(y)+1) = pre_error;
        %添加新元素
    end
end

%最后反差分编码
y(2:end) = -y(2:end);
for i = 2:length(y)
    y(i) = y(i)+y(i-1);
end

end

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            