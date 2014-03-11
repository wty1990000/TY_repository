
F_B_NEW = zeros(size(F_B));

for i = 1 : size(F_B)
    if(size(F_B{i}) ==0)
        F_B_NEW(i) = -1;
    else
        F_B_NEW(i) = F_B{i};
    end
end

COMB_UV = [A, F_B_NEW];