#include "DLList.h"
#include "TString.h"
#include "Editor.h"

using namespace std;

int main(int argc, char *argv[])
{
	if(argc != 3){
		cout<<"Usage:\n\t edit inputfile outputfile"<<endl;
		exit(1);
	}
	ifstream file_in(argv[1]);
	if(file_in ==0){
		cout<<"Can't open input file"<<argv[1]<<endl;
		exit(1);
	}
	ofstream file_out(argv[2]);
	if(file_out ==0){
		cout<<"Can't open output file"<<argv[2]<<endl;
		exit(1);
	}
	Editor buffer(&file_in, &file_out);
	while(buffer.get_command())
		buffer.run_command();
}