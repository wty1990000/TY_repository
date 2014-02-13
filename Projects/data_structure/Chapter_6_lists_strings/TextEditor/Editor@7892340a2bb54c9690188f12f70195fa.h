#ifndef EDITOR_H_
#define EDITOR_H_

#include "DLList.h"
#include "TString.h"

class Editor:public DLList<TString>{
public:
	Editor(std::ifstream *file_in, std::ofstream *file_out);
	bool get_command();
	void run_command();
private:
	std::ifstream *infile;
	std::ofstream *outfile;
	char user_command;

	Error_code next_line();
	Error_code previous_line();
	Error_code goto_line();
	Error_code insert_line();
};

#endif // !EDITOR_H_H
