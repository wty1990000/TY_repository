% fid = fopen('user_ID.txt','wt');
fid = fopen('A.txt','wt');
fid2 = fopen('venues.txt','wt');

% for i = 1:size(ID_U)
%         for j = 1: size(ID_U{i})
%             fprintf(fid,'%d,',ID_U{i}(j));
%         end
%         fprintf(fid,'\n');
% end

for i = 1:size(A)
    fprintf(fid,'%d,\n',A(i));
end
for j = 1:size(F_V)
    fprintf(fid2,'%d,\n',F_V(j));
end


% fclose(fid);
fclose(fid);
fclose(fid2);