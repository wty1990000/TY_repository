#include "Random.h"
#include <ctime>
#include <limits>
#include <cmath>

int Random::reseed()
/*Post: The seed is replaced by a pseudorandom successor.*/
{
	seed = seed*multiplier + add_on;
	return seed;
}

Random::Random(bool pseudo)
{
	if(pseudo) seed = 1;
	else seed = time(NULL)%INT_MAX;
	multiplier = 2743;
	add_on = 5923;
}

double Random::random_real()
/*Post: A random real number between [0,1) is returned*/
{
	double max = INT_MAX + 1.0;
	double temp = reseed();
	if(temp<0) temp = temp + max;
	return temp/max;
}
int Random::random_integer(int low, int high)
/*Post: A random integer number between [low, high)*/
{
	if(low>high) return random_integer(high,low);
	else return((int)((high-low+1)*random_real()))+low;
}
int Random::poisson(double mean)
/*Post: A random poisoson number with average mean.*/
{
	double limit = exp(-mean);
	double product = random_real();
	int count = 0;
	while(product>limit){
		count++;
		product *= random_real();
	}
	return count;
}