#ifndef LSTACK_H_
#define LSTACK_H_

#include "Node.h"
#include "utilities.h"

class LStack{
public:
	LStack();
	~LStack();
	LStack& operator=(const LStack &original);
	bool empty() const;
	Error_code push(const Stack_entry &item);
	Error_code pop();
	Error_code top(Stack_entry &item)const;
protected:
	Node *top_node;
};

#endif // !LSTACK_H_
