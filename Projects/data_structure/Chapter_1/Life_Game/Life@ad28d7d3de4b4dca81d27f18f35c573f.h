#ifndef _LIFE_H
#define _LIEF_H

const int maxrow = 20, maxcol = 20;

class Life{
public:
	void initialize();
	void upadate();
	void print();
private:
	int grid[maxrow+2][maxcol+2];
	int neighbor_count(int row, int col);
};

#endif // !_LIFE_H

