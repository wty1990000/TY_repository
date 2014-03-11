<Format for checking_local_dedup>
userID, venueID, timestamp

<Format for ISS755_checkin_small>
userID, venueID, timestamp, venue category, latitude, lontutide

<venue_info>
venueID, fine category, lat, lon

<TODO>
map venueID to unique integers, start from 0
weight is value 0.0 to 1.0
Jaccard similarity, i.e. proportion of common checkin users

Following files required
(1) edge file
venueID  venueID  weight


(2) mapping 1
venueID   coarse-category-id

(3) mapping 2
coarse-category-id   category

(refer http://aboutfoursquare.com/foursquare-categories)
