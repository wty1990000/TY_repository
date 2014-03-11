l=1;
fid = fopen('ISC_full.txt','wt');
% fid1 = fopen('ISC_full.txt','wt');

F_A = A;

for i= (43107-1):-1:1
    for m=l:l-1+i
        tempV = F_V(43107-i);
        tempV2 = F_V(43107-i+m-l+1);
        Numer = 0;
        tempID1 = ID_U{43107-i+m-l+1};
        tempID = ID_U{43107-i};
        for j = 1:size(tempID1)
            for p = 1:size(tempID)
                if(F_A(tempID1(j)) == F_A(tempID(p)))
                    Numer = Numer+1;
                end
            end
        end
        weight = Numer/(NumofCheckin(43107-i)+NumofCheckin(43107-i+m-l+1));
        if weight >= 0.2
           fprintf(fid,'%d,%d,%f,\n',tempV,tempV2,weight);
        end
%         fprintf(fid1,'%d,%d,%f,\n',tempV,tempV2,weight);
    end
    l = l+i;
end
fclose(fid);
% fclose(fid1);
