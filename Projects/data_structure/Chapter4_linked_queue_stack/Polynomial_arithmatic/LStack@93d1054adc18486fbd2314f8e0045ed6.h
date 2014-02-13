#ifndef LSTACK_H_
#define LSTACK_H_

typedef Stack_entry Node_entry;

#include "Node.h"
#include "utilities.h"

class LStack{
public:
	LStack();
	LStack(const LStack &original);				//Copy constructor
	~LStack();									//Destructor
	LStack& operator=(const LStack &original);	//Overloaded assignment
	bool empty() const;
	Error_code push(const Stack_entry &item);
	Error_code pop();
	Error_code top(Stack_entry &item)const;
	void print_stack()const;
protected:
	Node *top_node;
};

#endif // !LSTACK_H_
