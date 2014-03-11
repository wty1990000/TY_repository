%Prerequisit: C, F_C, F_V, VV, iV, res, res2

F_LAT = cell(size(VV));
F_LONG = cell(size(VV));

lat = res((1:end),3);
long = res((1:end),4);

for i= 1: size(iV)
    F_LAT{i} = lat{iV(i)};
    F_LONG{i} = long{iV(i)};
end

F_LAT = cell2mat(F_LAT);
F_LONG = cell2mat(F_LONG);

fid = fopen('venue_with_lat_long.txt','wt');

for i = 1:size(F_V)
    fprintf(fid, '%d,%d,%s,%f,%f,\n', F_V(i),Cat_ID(i),F_C{i}, F_LAT(i), F_LONG(i));
end

fclose(fid);