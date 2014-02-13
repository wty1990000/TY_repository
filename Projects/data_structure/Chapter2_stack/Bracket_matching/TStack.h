#ifndef TSTACK_H
#define TSTACK_H	

#include "utilities.h"

const int maxstack = 10;		//small value for testing

typedef char stack_entry;

class TStack{
public:
	TStack();
	bool empty() const;
	Error_code pop();
	Error_code top(stack_entry &item) const;
	Error_code push(const stack_entry &item);
private:
	int count;
	stack_entry entry[maxstack];
};

#endif // !TSTACK_H


