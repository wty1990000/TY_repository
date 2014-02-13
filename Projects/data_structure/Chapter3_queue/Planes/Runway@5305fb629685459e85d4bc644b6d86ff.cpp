#include "Runway.h"

using namespace std;

Runway::Runway(int limit)
/*Post: Initialize the status of the runway*/
{
	queue_limit = limit;
	num_land_requests = num_takeoff_requests = 0;
	num_landings = num_takingoff = 0;
	num_land_refused = num_takeoff_refused = 0;
	num_land_accepted = num_takeoff_accepted =0;
	land_wait = takeoff_wait = idle_time = 0;
}
Error_code Runway::can_land(const Plane &current)
/*Post: If possible, the Plane current is added to the land queue; otherwise
		Error_code overflow is returned.
  Uses: class Extended_queue*/
{
	Error_code result;
	if(landing.size()<queue_limit)
		result = landing.append(current);
	else
		result = fail;
	num_land_requests++;
	if(result != success)
		num_land_refused++;
	else
		num_land_accepted++;
	return result;
}
Error_code Runway::can_depart(const Plane &current)
/*Post: If possible, the Plance current is added to the depart queue; otherwise 
		Error_code overflow is returned. Update the Runway status.
  Uses: class Extended_queue*/
{
	Error_code result;
	if(takeoff.size()<queue_limit)
		result = takeoff.append(current);
	else
		result = fail;
	num_takeoff_requests++;
	if(result != success)
		num_takeoff_refused++;
	else
		num_takeoff_accepted++;
	return result;
}
Runway_activity Runway::activity(int time, Plane &moving)
/*Post: If landing queue has entries, the front plane is copied to the moving, a result of
		landing is returned; otherwise,if taking-off queue has entries, the front plane is
		copied to the moving, and a result of take off is returned; Otherwise idle is returned.
		Update the Runway status.*/
/*Uses: class Extended_queue*/
{
	Runway_activity in_progress;
	if(!landing.empty()){
		landing.retrieve(moving);
		land_wait += time-moving.started();
		num_landings++;
		in_progress = land;
		landing.serve();
	}
	else if(!takeoff.empty()){
		takeoff.retrieve(moving);
		takeoff_wait += time-moving.started();
		num_takingoff++;
		in_progress = takeoffs;
		takeoff.serve();
	}
	else{
		idle_time++;
		in_progress = idle;
	}
	return in_progress;		
}
void Runway::shut_down(int time)const
/*Post: Runway status are summerized and printed.*/
{
	cout<<"Simulation has conluded after "<<time<<" time units."<<endl
		<<"Total number of planes processed "
		<<(num_land_requests+num_takeoff_requests)<<endl
		<<"Total number of planes asking to land "
		<<num_land_requests<<endl
		<<"Total number of planes asking to take off "
		<<num_takeoff_requests<<endl
		<<"Total number of planes accepted for landing "
		<<num_land_accepted<<endl
		<<"Total number of planes accepted for taking off "
		<<num_takeoff_accepted<<endl
		<<"Total number of planes refused for landing "
		<<num_land_refused<<endl
		<<"Total number of planes refused for takeoff "
		<<num_takeoff_refused<<endl
		<<"Total number of planes that landed "
		<<num_landings<<endl
		<<"Total number of planes that took off "
		<<num_takingoff<<endl
		<<"Total number of planes left in landing queue "
		<<landing.size()<<endl
		<<"Total number of planes left in takeoff queue "
		<<takeoff.size()<<endl;
	cout<<"Percetage of tiem runway idle "
		<<100.0*((float)idle_time)/((float)time)<<"%"<<endl;
}