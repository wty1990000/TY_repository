#ifndef _TIMER_H
#define _TIMER_H

#include <ctime>

class Timer{
public:
	Timer();
	double elapsedTime();
	void reset();
private:
	clock_t startTime;
};


#endif // !_TIMER_H
