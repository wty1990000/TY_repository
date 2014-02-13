#include "Polynomials.h"

using namespace std;

void Polynomial::print()const
{
	Node *print_node = front;
	bool first_term = true;
	while(print_node != nullptr){
		Term &print_term = print_node->entry;
		if(first_term){
			first_term = false;
			if(print_term.coefficient<0) cout<<" - ";
		}
		else if(print_term.coefficient<0) cout<<" - ";
		else cout<<" + ";
		double r =(print_term.coefficient>=0)?print_term.coefficient:-(print_term.coefficient);
		if(r!=1) cout<<r;
		if(print_term.degree>1) cout<<" X^"<<print_term.degree;

	}
}