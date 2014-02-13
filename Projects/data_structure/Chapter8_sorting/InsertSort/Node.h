#ifndef NODE_H_
#define NODE_H_

#include "utilities.h"

template<class Node_entry>
struct Node{
	//data members
	Node_entry entry;
	Node<Node_entry> *next;
	//constructors
	Node();
	Node(Node_entry item,Node<Node_entry> *link = nullptr);
};

template<class Node_entry>
Node<Node_entry>::Node()
{
	next = nullptr;
}
template<class Node_entry>
Node<Node_entry>::Node(Node_entry item, Node<Node_entry> *link)
{
	entry = item;
	next = link;
}

#endif // !NODE_H_
