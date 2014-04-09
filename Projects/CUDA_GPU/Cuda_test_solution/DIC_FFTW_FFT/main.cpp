#include <iostream>
#include "fftw3.h"
#include "Random.h"
#include "helper_timer.h"

#include "loop_plan.h"
#include "batch_plan.h"

int main()
{
	StopWatchWin loop_plan, batch_plan;
	float loop_time, batch_time, ttother, tother;

	loop_plan.start();
		loopplan(ttother);
	loop_plan.stop();
	loop_time =ttother;
	
	batch_plan.start();
		batchedplan(tother);
	batch_plan.stop();
	batch_time = tother;


	std::cout<<"Loop FFTs consume:"<<loop_time<<std::endl;
	std::cout<<"Batch FFTs consume:"<<batch_time<<std::endl;

	return 0;
}