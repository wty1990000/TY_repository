#ifndef RANDOM_H_
#define RANDOM_H_


class Random{
public:
	Random(bool pseudo = true);
	// Declare random-number generation methods here.
	double random_real();
	int random_integer(int low, int high);
	int poisson(double mean);
private:
	int reseed();  //Re-randomize the seed.
	int seed, multiplier, add_on;
};

#endif // !RANDOM_H_
