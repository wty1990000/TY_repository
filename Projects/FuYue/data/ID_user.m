ID_U = cell(size(F_V));

for ii = 1:size(F_V)
    ID_U{ii} = find(COMB_UV(:,2)==F_V(ii));
end