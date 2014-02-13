#include "TQueue.h"
#include "Extended_queue.h"

using namespace std;

void help()
/*Post: print the help screen*/
{
	cout<<endl
		<<"This program allows user to enter one command"<<endl
		<<"(but only one) on each input line."<<endl
		<<"For example, if the command S is entered, then"<<endl
		<<"the program will serve the front of the queue."<<endl
		<<endl
		<<" The valid commands are:"<<endl
		<<"A - Append the next input character to the extended queue"<<endl
		<<"S - Serve the front of the extended queue"<<endl
		<<"R - Retrive and print the front entry."<<endl
		<<"# - The current size of the extended queue"<<endl
		<<"C - Clear the extended queue(same as delete)"<<endl
		<<"P - Print the extended queue"<<endl
		<<"H - This help screen"<<endl
		<<"Q - Quit"<<endl
		<<"Press <Enter> to continue."<<flush;
	char c;
	do
	{
		cin.get(c);
	}while(c!='\n');
}
void intruduction()
{
	cout<<"This is a menu-driven program to demonstrate the queue structure."<<endl;
}
char get_command()
/*Post: return the command to the program*/
{
	char command;
	bool waiting = true;
	while(waiting){
		cin>>command;
		command = tolower(command);
		if(command == 'a' || command == 's' || command == 'r' ||
			command =='#' || command == 'c' || command == 'p' ||
			command =='p' || command == 'h' || command == 'q')
			waiting = false;
		else{
			cout<<"Plaase enter a valid command:"<<endl
				<<" The valid commands are:"<<endl
				<<"A - Append the next input character to the extended queue"<<endl
				<<"S - Serve the front of the extended queue"<<endl
				<<"R - Retrive and print the front entry."<<endl
				<<"# - The current size of the extended queue"<<endl
				<<"C - Clear the extended queue(same as delete)"<<endl
				<<"P - Print the extended queue"<<endl
				<<"H - This help screen"<<endl
				<<"Q - Quit"<<endl;
		}
	}
	return command;
}
bool do_command(char c, Extended_queue &test_queue)
/*Pre: C represents appropriate 
  Post: Do the appropirate command acoording to the value of C*/
{
	bool continue_input = true;
	Queue_entry x;
	switch(c){
	case 'q':
		cout<<"Extended queue demonstration finished."<<endl;
		continue_input = false;
		break;
	case 'a':
		cin>>x;
		if(test_queue.append(x) == overflow)
			cout<<"Queue is full."<<endl;
		else
			cout<<endl<<"The entry "<<x<<"is successfully appended"<<endl;
		break;
	case 's':
		if(test_queue.serve() == underflow)
			cout<<"Queue is empty."<<endl;
		else
			cout<<endl<<"The front entry is successfully removed."<<endl;
		break;
	case 'r':
		if(test_queue.retrieve(x) == underflow)
			cout<<"Queue is empty."<<endl;
		else
			cout<<endl<<"The first entry is: "<<x<<endl;
		break;
	case '#':
		cout<<endl<<test_queue.size()<<endl;
	case 'c':
		test_queue.clear();
		cout<<endl<<"The queue is cleared."<<endl;
		break;
	case 'p':
		test_queue.print_queue();
		break;
	case 'h':
		help();
		
	}
	
	return continue_input;
}

int main()
/*Pre: Accept commands from user and do the corresponding operations.
  Uses: TQueue, Extended_queue, help, instruction, get_command, do_command
*/
{
	Extended_queue test_queue;
	intruduction();
	help();
	while(do_command(get_command(),test_queue));
}