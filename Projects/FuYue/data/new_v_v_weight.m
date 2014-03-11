l=1;
fid = fopen('ISC_full.txt','wt');
% fid1 = fopen('ISC_full_full.txt','wt');

F_A = A;


for i= (43107-1):-1:1
    for m=l:l-1+i
        tempV = F_V(43107-i);
        tempV2 = F_V(43107-i+m-l+1);
        Numer = 0;
        tempID2 = ID_U{43107-i+m-l+1};
        tempID1 = ID_U{43107-i};
%         for j = 1:size(tempID1)
%             for p = 1:size(tempID)
%                 if(F_A(tempID1(j)) == F_A(tempID(p)))
%                     Numer = Numer+1;
%                 end
%             end
%         end 
        tempFA1 = zeros(size(tempID1));
        tempFA2 = zeros(size(tempID2));
        for j= 1:size(tempID1)
            tempFA1(j) = F_A(tempID1(j));
        end
        for p = 1:size(tempID2)
            tempFA2(p) = F_A(tempID2(p));
        end
        
        [tempFA11, iFA1, itFA1] = unique(tempFA1,'stable');
        [tempFA22, iFA2, itFA2] = unique(tempFA2,'stable');
        
        for jj = 1: size(tempFA22)
            TTT = find(tempFA11 == tempFA22(jj));
        end
        Numer = length(TTT);
%         for jj=1:size(tempFA11)
%             for pp = 1:size(tempFA22)
%                 if(tempFA11(jj) == tempFA22(pp))
%                     Numer = Numer+1;
%                 end
%             end
%         end
        weight = Numer/(NumofCheckin(43107-i)+NumofCheckin(43107-i+m-l+1)-Numer);
        if weight >= 0.01
           fprintf(fid,'%d,%d,%f\n',tempV,tempV2,weight);
        end
%         fprintf(fid1,'%d,%d,%f,\n',tempV,tempV2,weight);
    end
    l = l+i;
end
fclose(fid);
% fclose(fid1);
