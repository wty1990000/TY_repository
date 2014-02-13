#include "Plane.h"

using namespace std;

Plane::Plane()
/*Post: The plane data members are set to illegal default values*/
{
	flt_num = -1;
	clock_start = -1;
	state = null;
}
Plane::Plane(int flt, int time, Plane_status status)
	:flt_num(flt), clock_start(time), state(status)
/*Post: The Plane data members are set to user-defined values.*/
{
	cout<<"Plane number "<<flt<<" ready to ";
	if(status == arriving)
		cout<<"land."<<endl;
	else
		cout<<"take off."<<endl;
}
void Plane::refuse() const
/*Post: refuse a Plane wanting to use Runway, when queue is full*/
{
	cout<<"Plane number "<<flt_num;
	if(state == arriving)
		cout<<" directed to another airport"<<endl;
	else
		cout<<" told to try to takeoff again later"<<endl;
}
void Plane::land(int time) const
/*Post: Process landing planes*/
{
	int wait = time - clock_start;
	cout<<time<<": Plane number "<<flt_num<<" landed after "
		<<wait<<" time unit"<<((wait ==1)?"" :"s")
		<<"in the landing queue."<<endl;
}
void Plane::fly(int time)const
/*Post: Process taking-off planes*/
{}