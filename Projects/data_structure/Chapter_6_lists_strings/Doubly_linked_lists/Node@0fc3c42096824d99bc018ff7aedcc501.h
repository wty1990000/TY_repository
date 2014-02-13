#ifndef NODE_H_
#define NODE_H_

#include "utilities.h"

template<class Node_entry>
struct Node{
	//data members
	Node_entry entry;
	Node<Node_entry> *next;
	Node<Node_entry> *back;
	//constructors
	Node();
	Node(Node_entry item, Node<Node_entry> *link_next = nullptr,
						  Node<Node_entry> *link_back = nullptr);
};

template<class Node_entry>
Node<Node_entry>::Node()
	:next(nullptr), back(nullptr)
{}
template<class Node_entry>
Node<Node_entry>::Node(Node_entry item, Node<Node_entry> *link_next,
										Node<Node_entry> *link_back)
	:entry(item),next(link_next),back(link_back)
{}

#endif // !NODE_H_
