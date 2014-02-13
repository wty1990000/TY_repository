#include "Node.h"

Node::Node()
{
	next = nullptr;
}
Node::Node(Node_entry item, Node *add_on)
{
	entry = item;
	next = add_on;
}
