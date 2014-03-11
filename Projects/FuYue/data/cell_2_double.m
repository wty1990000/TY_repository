F_BB = zeros(size(F_B));

for i = 1: size(F_B)
    if(size(F_B(i)) == 0)
        F_BB(i) =0;
    else
        F_BB(i) = F_B{i};
    end
end