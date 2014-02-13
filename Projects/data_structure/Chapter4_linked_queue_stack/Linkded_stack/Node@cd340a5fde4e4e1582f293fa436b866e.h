#ifndef NODE_H_
#define NODE_H_

struct Node{
	//data members
	Node *next;
	Node_entry entry;
	//constructors
	Node();
	Node(Node_entry item, Node *add_on = nullptr);
};

#endif // !NODE_H_

