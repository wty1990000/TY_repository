#include "Extended_queue.h"
#include "Plane.h"
#include "Random.h"
#include "Runway.h"
#include "TQueue.h"
#include "utilities.h"

using namespace std;

void initialize(int &end_time, int &queue_limit, double &arrival_rate, double &departure_rate)
/*Pre: Use specifies the number of time units, the maximum queue limit, the arrival rate and the 
		departure rate.
  Post: The program print instructions and initialize the parameters.*/
{
	cout<<"This program simulates an airport with only one runway."<<endl
		<<"One plane can land or depart in each unit of time."<<endl;
	cout<<"Up to waht numberof panles can be waiting to land "
		<<"or take off at any time?"<<flush;
	cin>>queue_limit;
	cout<<"How many units of time will the simulation run?"<<flush;
	cin>>end_time;
	bool acceptable;
	do{
		cout<<"Expected number of arrivals per unit time?"<<flush;
		cin>>arrival_rate;
		cout<<"Expected number of departures per unit time?"<<flush;
		cin>>departure_rate;
		if(arrival_rate<0.0 || departure_rate<0.0)
			cerr<<"These rates must be nonnegative."<<endl;
		else
			acceptable = true;
		if(acceptable && arrival_rate+departure_rate>1.0)
			cerr<<"Safety warning: This airport will become saturated."<<endl;
	}while(!acceptable);
}
void run_idle(int time)
/*Post: the specified time is printed with a messhage that the runway is idle.*/
{
	cout<<time<<": Runway is idle."<<endl;
}
int main()
/*Pre: Number of time intervals to run the simulation, the expected
		number of planes arriving, departing per time interval, maximum
		allowed size for the runway queues.
  Post: Showing the status of runways per time interval.
  Uses: Classes Runway, Plane, Random and functions run_idle, initialize*/
{
	int end_time;
	int queue_limit;
	int flight_number = 0;
	double arrival_rate, departure_rate;
	initialize(end_time,queue_limit,arrival_rate,departure_rate);
	Random variable;
	Runway small_airport(queue_limit);
	for(int current_time =0; current_time<end_time; current_time++){
		int number_arrivals = variable.poisson(arrival_rate);
		for(int i=0; i<number_arrivals; i++){
			Plane current_plane(flight_number++,current_time,arriving);
			if(small_airport.can_land(current_plane)!=success)
				current_plane.refuse();
		}
		int number_departures = variable.poisson(departure_rate);
		for(int j=0; j<number_departures; j++){
			Plane current_plane(flight_number++,current_time,departing);
			if(small_airport.can_depart(current_plane)!=success)
				current_plane.refuse();
		}
		Plane moving_plane;
		switch(small_airport.activity(current_time,moving_plane)){
		case land:
			moving_plane.land(current_time);
			break;
		case takeoffs:
			moving_plane.fly(current_time);
			break;
		case idle:
			run_idle(current_time);
		}
	}
}