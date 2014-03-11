res3 = csv2cell('VenueID_CategoryID.csv');

Cat_ID = res3((1:end),3);
Cat_ID = cell2mat(Cat_ID);