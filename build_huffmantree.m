function tree = build_huffmantree(encode_table)
%根据ACTAB，DCTAB表来构建huffmantree
%encode_table should be this form：
%
%
%[numbers of value； value； numbers of code， code]
% and I will return a struct tree
% 
% each node has 2 fields called 'left' and 'right',their values
% are index of the child


tree = struct('left',0,'right',0,'value',-1);     %根节点

[r,c] = size(encode_table);
for i = 1:r         %共有r个编码
    row = encode_table(i,:);
    value_number = row(1);              %值的个数
    code_number = row(2+value_number);  %编码长度
    index = 1;                          
    code = row(3+value_number:3+value_number+code_number-1); %获取code
    
    for k = 1:length(code)      %对tree进行构造
        if(code(k)==0)
            if(tree(index).left~=0)         %若存在left节点
                index = tree(index).left;   %读取下一个节点index
            else                          
                tree(length(tree)+1) =  struct('left',0,'right',0,'value',-1);     %创建新节点
                tree(index).left = length(tree);              %对left进行赋值
                index = length(tree);
                %disp(strcat('create a new node, index:',num2str(index)));
            end
        elseif(code(k)==1)
            if(tree(index).right~=0)        %若存在right节点
                index = tree(index).right;   %读取下一个节点index
            else                          
                tree(length(tree)+1) =  struct('left',0,'right',0,'value',-1);      %创建新节点
                tree(index).right = length(tree);  %创建新节点
                index = length(tree);
            end
        else
            error('code should not contain numbers otherwise 1,0');
        end
    end
    tree(index).value = row(2:value_number+1);      %定义节点的value即为解码结果
end
