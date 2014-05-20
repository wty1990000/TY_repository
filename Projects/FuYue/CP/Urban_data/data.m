clear all 
close all

C = csv2cell('urban_pop.csv');
C1 = C(:,1);

fid = fopen('Urban_data.csv', 'wt');

i=1960;

fprintf(fid,'id,Year,cases\n');

for j = 2:1:51
    Co = C(:,j);
    Co = cell2mat(Co);
    for m = 1:1:206
        fprintf(fid,'%s,%d,%3.2f\n',C1{m},i,Co(m));
    end
    i = i+1;
end

fclose(fid);