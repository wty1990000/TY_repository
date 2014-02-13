#ifndef TSTACK_H
#define TSTACK_H	

#include "utilities.h"

const int maxstack = 10;		//small value for testing

class TStack{
public:
	TStack();
	bool empty() const;
	Error_code pop();
	Error_code top(stack_entry &item) const;

};

#endif // !TSTACK_H


