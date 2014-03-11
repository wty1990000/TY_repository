clear all
close all

res = csv2cell('venue_info.txt');
res2 = csv2cell('checking_local_dedup.txt');

V = res((1:end),1);
C = res((1:end),2);

B = res2((1:end),2);
A = res2((1:end),1);

[VV, iV, iVV]= unique(V,'stable');

F_C = cell(size(VV));

for i = 1:size(iV)
    F_C{i} = C{iV(i)};
end


F_V = cell(size(VV));
F_B = cell(size(B));
% 
NumofCheckin = zeros(size(VV)); %??venue?checkin??

for m=1:size(VV)
    F_V{m} = m-1;
end

A = cell2mat(A);

%Find the data of ISC matching venue_infor indices
for i = 1: length(VV)
    Q = find(strcmp(B,VV{i}));
    NumofCheckin(i) = length(Q);
    for j = 1 : length(Q)
        F_B{Q(j)} = F_V{i};
    end
end

% COMB = [A,F_B];
% COMB_UV = cell2mat(COMB);%COMBINED User and Venue ID correspondence
COMB_VC = [F_V, F_C];

F_V = cell2mat(F_V);



