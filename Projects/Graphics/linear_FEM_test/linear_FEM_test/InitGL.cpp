#include "InitGL.h"

using namespace std;

extern void displayFunc(void);
extern void reshapeFunc(int iWindowWidth, int iWindowHeight);
extern void idleFunc(void);
extern void keyboardFunc(unsigned char key, int x, int y);
extern void onMouseDown(int button, int state, int x, int y);
extern void onMouseMotion(int x, int y);

/* --------- OpenGL initialization functions -----------*/
void initGL( int iWindowWidth, int iWindowHeight)
{
	graphicalglobals = new GraphicGlobalVariables();
	glClearColor(148.0f / 256, 199.0f/256, 211.0f/256, 0.0);

	glEnable(GL_DEPTH_TEST);
	glShadeModel(GL_SMOOTH);
	glEnable(GL_POLYGON_SMOOTH);
	glEnable(GL_LINE_SMOOTH);

	reshapeFunc(iWindowWidth, iWindowHeight);
	cout<<"Graphics intitialization has completed."<<endl;
}
void initGlut(int argc, char **argv, char *pt_cWindowTitle, 
			  int iWindowWidth, int iWindowHeight, int *pt_iWindowId)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(iWindowWidth, iWindowHeight);
	*pt_iWindowId = glutCreateWindow(pt_cWindowTitle);
	
	glutDisplayFunc(displayFunc);
	glutReshapeFunc(reshapeFunc);
	glutIdleFunc(idleFunc);
	glutKeyboardFunc(keyboardFunc);
	glutMouseFunc(onMouseDown);
	glutMotionFunc(onMouseMotion);
}

/* --------- Draw the environment -----------*/
void drawGround()
{
	glBegin(GL_LINES);
		glColor3f(0.5f, 0.5f, 0.5f);
		for(int i = -iGridSize; i <= iGridSize; i++){
			glVertex3f((float)i, 0, (float)-iGridSize);
			glVertex3f((float)i, 0, (float)iGridSize);

			glVertex3f((float)-iGridSize, 0, (float)i);
			glVertex3f((float)iGridSize, 0, (float)i);
		}
	glEnd();
}
void drawAxis()
{
	glBegin(GL_LINES);
		for( int i=0; i<3; i++){
			float fColor[3] = {0.0f, 0.0f, 0.0f};
			fColor[i] = 1.0;
			glColor3fv(fColor);

			float iVertex[3] = {0.0f, 0.0f, 0.0f};
			iVertex[i] = 5.0f;
			glVertex3fv(iVertex);
			glVertex3f(0.0, 0.0, 0.0);
		}
	glEnd();
}