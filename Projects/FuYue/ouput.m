% fid = fopen('user_ID.txt','wt');
fid = fopen('venues.txt','wt');
% fid2 = fopen('venues.txt','wt');

% for i = 1:size(ID_U)
% %     if isempty(ID_U{i})
% %         fprintf(fid,'%d,\n',-1);
% %     else
%         for j = 1: size(ID_U{i})
%             fprintf(fid,'%d,',ID_U{i}(j));
%         end
%         fprintf(fid,'\n');
% %     end
% end



for i = 1:size(F_B_NEW)
    fprintf(fid,'%d,\n',F_B_NEW(i));
end
% for j = 1:size(F_V)
%     fprintf(fid2,'%d,\n',F_V(j));
% end
% 

% fclose(fid);
fclose(fid);
% fclose(fid2);