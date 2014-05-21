clear all 
close all

C = csv2cell('largest_city_pop.csv');
C1 = C(:,1);

fid = fopen('large_data.csv', 'wt');

i=1960;

fprintf(fid,'id,Year,cases\n');

for j = 2:1:51
    Co = C(:,j);
    Co = cell2mat(Co);
    for m = 1:1:size(C1)
        fprintf(fid,'%s,%d,%f\n',C1{m},i,Co(m));
    end
    i = i+1;
end

fclose(fid);