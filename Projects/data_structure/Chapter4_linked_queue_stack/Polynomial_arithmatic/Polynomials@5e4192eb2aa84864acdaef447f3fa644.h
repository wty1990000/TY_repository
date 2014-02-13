#ifndef POLYNOMIALS_H_
#define POLYNOMIALS_H_

#include "Extended_queue.h"

class Polynomial:private Extended_queue{
public:
	void read();
	void print()const;
	void equal_sum(Polynomial p, Polynomial q);
	void equal_difference(Polynomial p, Polynomial q);
	void equal_product(Polynomial p, Polynomial q);
	Error_code equals_quotient(Polynomial p, Polynomial q);
	int degree()const;
private:
	void mul_term(Polynomial p, Term t)
};

#endif // !POLYNOMIALS_H_
