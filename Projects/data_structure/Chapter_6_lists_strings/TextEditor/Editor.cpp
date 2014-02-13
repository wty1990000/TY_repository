#include "Editor.h"

using namespace std;

Editor::Editor(ifstream *file_in, ofstream *file_out)
	:infile(file_in),outfile(file_out)
{}
bool Editor::get_command()
{
	if(current !=nullptr)
		cout<<current_position<<" : "
			<<current->entry.c_str()<<"\n??"<<flush;
	else
		cout<<"File is empty.\n??"<<flush;
	cin>>user_command;
	user_command = tolower(user_command);
	while(cin.get()!='\n')
		;
	if(user_command =='q')
		return false;
	else
		return true;
}
void Editor::run_command()
{
	TString temp_string;
	switch (user_command)
	{
	case 'b':
		if(empty())
			cout<<"Warning: empty buffer"<<endl;
		else
			while(previous_line() == success)
				;
		break;
	case 'c':
		if(empty())
			cout<<"Warning: empter file"<<endl;
		else if(change_line()!=success)
			cout<<"Error: Substitution failed"<<endl;
		break;
	case 'd':
		if(remove(current_position,temp_string)!=success)
			cout<<"Error: Deletion failed"<<endl;
		break;
	case 'e':
		if(empty())
			cout<<"Warning:empty buffer"<<endl;
		else
			while(next_line()==success)
				;
		break;
	case 'f':
		if(empty())
			cout<<"Warning:empty file"<<endl;
		else
			find_string();
		break;
	case 'g':
		if(goto_line()!=success)
			cout<<"Warning:No such line"<<endl;
		break;
	case '?':
	case 'h':
		cout<<"Valid commands are: b(egin) c(hange) d(el) e(nd)"<<endl
			<<"f(ind) g(o) h(elp) i(nsert) l(ength) n(ext) p(rior)"<<endl
			<<"q(uit) r(ead) s(ubstitute) v(iew) w(rite)"<<endl;
	case 'i':
		if(insert_line()!= success)
			cout<<"Error: Insertion failed"<<endl;
		break;
	case 'l':
		cout<<"There are "<<size()<<" lines in the file."<<endl;
		if(!empty())
			cout<<"Current line length is: "
				<<strlen((current->entry).c_str())<<endl;
		break;
	case 'n':
		if(next_line() != success)
			cout<<"Warning: at the end of buffer"<<endl;
		break;
	case 'p':
		if(previous_line() != success)
			cout<<"Warning: at the start of buffer"<<endl;
		break;
	case 'r':
		read_file();
		break;
	case 's':
		if(substitue_line()!=success)
			cout<<"Error: Substitution failed"<<endl;
		break;
	case 'v':
		traverse(write);
		break;
	case 'w':
		if(empty())
			cout<<"Warning: Empty file"<<endl;
		else
			write_file();
		break;
	default:
		cout<< "Press h or ? for help or enter a valid command:";
	}
}

void Editor::read_file()
{
	bool proceed = true;
	if(!empty()){
		cout<<"Buffer is not empty; the read will destroy it."<<endl;
		cout<<" OK to proceed?"<<endl;
		if(proceed = user_say_yes()) clear();
	}
	int line_number = 0, terminal_char;
	while(proceed){
		TString in_string = read_in(*infile,terminal_char)
	}
}