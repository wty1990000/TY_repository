#ifndef TERM_H_
#define TERM_H_

struct Term
{
	int degree;
	double coefficient;
	Term(int exponent = 0,double scalar = 0);
};

Term::Term(int exponent, double scalar)
{
	degree = exponent;
	coefficient = scalar;
}

#endif // !TERM_H_
