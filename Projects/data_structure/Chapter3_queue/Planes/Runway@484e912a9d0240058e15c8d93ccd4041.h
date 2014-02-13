#ifndef RUNWAY_H_
#define RUNWAY_H_

#include "utilities.h"
#include "Plane.h"
#include "Extended_queue.h"

enum Runway_activity
{
	idle, land, takeoff
};

class Runway{
public:
	Runway(int limit);
	Error_code can_land(const Plane &current);
	Error_code can_depart(const Plane &current);
	Runway_activity activity(int time, Plane &moving);
	void shut_down(int time) const;
private:
	Extended_queue landing;
	Extended_queue takeoff;
	int queue_limit;
	int num_land_requests;		//number of planes ask for landing 
	int num_takeoff_requests;	//number of planes ask for taking off
	int num_landings;			//number of planes have landed
	int num_takingoff;			//number of planes have taken off
	int num_land_accepted;		//number of planes queued to land
	int num_takeoff_accepted;	//number of planes queued to take off
	int num_land_refused;		//number of planes refused to land
	int num_takeoff_refused;	//number of planes refused to take off
	int land_wait;				//total time of planes waiting to land
	int takeoff_wait;			//total time of planes waiting to take off
	int idle_time;				//total time of idle
};

#endif // !RUNWAY_H_
