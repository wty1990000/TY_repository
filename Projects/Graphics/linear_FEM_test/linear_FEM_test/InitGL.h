#ifndef _INITGL_H_
#define _INITGL_H_

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <glm/glm.hpp>
#include "utilities.h"

class GraphicGlobalVariables
{	
public:
	GraphicGlobalVariables()
		: iOldX(0),iOldY(0),rX(15.0f),rY(0.0f),iState(1),fDist(-2.5f),Up(glm::vec3(0.0f,1.0f,0.0f))
	{}

	int iOldX, iOldY;
	float rX, rY;
	int iState;
	float fDist;
	glm::vec3 Up,Right,view;
	GLint iViewport[4];
	GLdouble dModelView[16];
	GLdouble dPorjection[16];

};
const int iGridSize = 10;

/* --------- OpenGL initialization functions -----------*/

void initGL( int iWindowWidth, int iWindowHeight);
void initGlut(int argc, char **argv, char *pt_cWindowTitle, 
			  int iWindowWidth, int iWindowHeight, int *pt_iWindowId);

/* --------- Draw the environment -----------*/
void drawGround();
void drawAxis();

#endif // !_INITGL_H_


