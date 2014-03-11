fid = fopen('user_venue_timestamp.txt','wt');

T = res2((1:end),3);

T = cell2mat(T);

for i=1:size(A)
    fprintf(fid,'%d,%d,%d,\n',A(i),F_B_NEW(i),T(i));
end

fclose(fid);