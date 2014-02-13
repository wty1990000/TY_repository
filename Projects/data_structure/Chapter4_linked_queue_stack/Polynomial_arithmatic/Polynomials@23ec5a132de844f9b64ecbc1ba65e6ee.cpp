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
			if(print_term.coefficient<0) cout<<"-";
		}
		else if(print_term.coefficient<0) cout<<" -";
		else cout<<" +";

	}
}