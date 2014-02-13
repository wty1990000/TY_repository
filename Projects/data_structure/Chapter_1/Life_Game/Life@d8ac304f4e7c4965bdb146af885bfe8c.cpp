#include "Life.h"
#include "utilities.h"

using namespace std;

void Life::initialize()
/*Pre: None
  Post: The Life object contains a configuration specified by the user*/
{
	int row, col;
	for(row=0; row<=maxrow+1; row++)
		for(col=0; col<maxcol+1; col++)
			grid[row][col] = 0;
	cout<<"List the coordinates for living cells."<<endl;
	cout<<"Terminate the list with the special pair -1 -1"<<endl;
	cin>>row>>col;
	while (row != -1 || col != -1)
	{
		if(row>=1 && row<=maxrow)
			if(col>=1 && col<=maxcol)
				grid[row][col] = 1;
			else
				cout<<"Column "<<col<<" is out of range."<<endl;
		else
			cout<<"Row "<<row<<" is out of range."<<endl;
		cin>>row>>col;
	}
}

int Life::neighbor_count(const int &row, const int &col)
/*Pre: The Life object constains a configuration and the coordinates row and col 
			define a cell inside its hedge
	  Post: The number of living neighbors of the specified cell is returned*/
{
	int count = 0;
	for(int i=row-1; i<=row+1; i++)
		for(int j = col-1; j<=col+1; j++)
			count += grid[i][j];		//increase the count if neighbor is alive
	count -= grid[row][col];			//Reduce count, since cell is not its own neighbor

	return count;
}

void Life::upadate()
/*Pre: The Life object contains a configuration
  Post: The Life object contains the next generation of configuration */
{
	int row,col;
	int new_grid[maxrow+2][maxcol+2];
	for(row=1; row<=maxrow; row++)
		for(col=1; col<=maxcol; col++)
			switch(neighbor_count(row,col)){
			case 2:
				new_grid[row][col] = grid[row][col];	//status stays the same
				break;
			case 3:
				new_grid[row][col] = 1;					//cell becomes alive again
				break;
			default:
				new_grid[row][col] = 0;					//cell dies
		}
}

void Life::print()
/*Pre: The Life contains a configuration
  Post: Output the configuration to the user*/
{
	int row, col;
	cout<<"\nThe current configuration is:"<<endl;
	for(row = 1; row <=maxrow; row++)
		for(col = 1; col <=maxcol; col++)
			if(grid[row][col] ==1)
				cout<<'*';
			else
				cout<<' ';
	cout<<endl;
}