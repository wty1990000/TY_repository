#ifndef TERM_H_
#define TERM_H_

struct Term
{
	int degree;
	double coefficient;
	Term(int exponent = 0,double scalar = 0);
};



#endif // !TERM_H_
