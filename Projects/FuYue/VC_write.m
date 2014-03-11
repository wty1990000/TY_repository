% load('COMB_VC.mat');

fid = fopen('COMB_V_C.csv','wt');

for i = 1:43107
        fprintf(fid,'%d,%s\n',COMB_VC{i,:});
end

fclose(fid);