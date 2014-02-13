typedef char Stack_entry;
typedef Stack_entry Node_entry;


struct Node{
	//data members
	Node *next;
	Node_entry entry;
	//constructors
	Node();
	Node(Node_entry item, Node *add_on = nullptr);
};