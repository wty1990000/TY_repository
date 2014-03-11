load('COMB_VC.mat');

fid = fopen('COMB_V_C.csv','wt');

for i = 1:8473
        fprintf(fid,'%d,%s\n',COMB_VC{i,:});
end

fclose(fid);