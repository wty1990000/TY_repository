#include "vgl.h"
#include <vmath.h>

#include "LoadShaders.h"
#include <iostream>

//#define USE_PRIMITIVE_RESTART 1

using namespace std;
using namespace vmath;

enum transform
{
	notransform,rotation
};

typedef struct global_t{
	float aspect;
	GLuint render_prog;
	GLuint vao[1];
	GLuint vbo[1];
	GLuint ebo[1];

	GLfloat angle;
	GLuint mode;
	GLint press_x, press_y;
	GLfloat x_angle, y_angle;

	GLint render_model_matrix_loc;
	GLint render_projection_matrix_loc;
}global;

global *glb;

void init()
{
	glb = new global();

	glb->x_angle = 0.0; glb->y_angle = 0.0;

	static ShaderInfo shader_info[] = 
	{
		{GL_VERTEX_SHADER, "vs.glsl"},
		{GL_FRAGMENT_SHADER, "fs.glsl"},
		{GL_NONE, NULL}
	};
	glb->render_prog = LoadShaders(shader_info);
	glUseProgram(glb->render_prog);

	glb->render_model_matrix_loc = glGetUniformLocation(glb->render_prog, "model_matrix");
	glb->render_projection_matrix_loc = glGetUniformLocation(glb->render_prog, "projection_matrix");

	//Cube vertices
	static const GLfloat cube_positions[] = 
	{
		-1.0f, -1.0f, -1.0f, 1.0f,
		-1.0f, -1.0f, 1.0f, 1.0f,
		-1.0f, 1.0f, -1.0f, 1.0f,
		-1.0f, 1.0f, 1.0f, 1.0f,
		1.0f, -1.0f, -1.0f, 1.0f,
		1.0f, -1.0f, 1.0f, 1.0f, 
		1.0f, 1.0f, -1.0f, 1.0f, 
		1.0f, 1.0f, 1.0f, 1.0f
	};

	//Vertex colors
	static const GLfloat cube_color[] = 
	{
		1.0f, 1.0f, 1.0f, 1.0f, 
		1.0f, 1.0f, 0.0f, 1.0f, 
		1.0f, 0.0f, 1.0f, 1.0f, 
		1.0f, 0.0f, 0.0f, 1.0f,
		0.0f, 1.0f, 1.0f, 1.0f, 
		0.0f, 1.0f, 0.0f, 1.0f,
		0.0f, 0.0f, 1.0f, 1.0f, 
		0.5f, 0.5f, 0.5f, 1.0f
	};

	//Element index
	static const GLushort cube_index[] = 
	{
		0, 1, 2, 3, 6, 7, 4, 5,
		0xFFFF,
		2, 6, 0, 4, 1, 5, 3, 7
	};

	//Set up element array buffer
	glGenBuffers(1, glb->ebo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, glb->ebo[0]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(cube_index), cube_index,GL_STATIC_DRAW);

	//Set up the vertex attributes
	glGenVertexArrays(1, glb->vao);
	glBindVertexArray(glb->vao[0]);

	glGenBuffers(1, glb->vbo);
	glBindBuffer(GL_ARRAY_BUFFER, glb->vbo[0]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(cube_positions) + sizeof(cube_color), NULL, GL_STATIC_DRAW);
	glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(cube_positions), cube_positions);
	glBufferSubData(GL_ARRAY_BUFFER,sizeof(cube_positions), sizeof(cube_color),cube_color);
	
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, NULL);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(sizeof(cube_positions)));
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);

	//glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
}
void display()
{
	//static float t = float(GetTickCount() & 0x1FFF) / float(0x1FFF);
	static const vmath::vec3 X(1.0f, 0.0f, 0.0f);
	static const vmath::vec3 Y(0.0f, 1.0f, 0.0f);
	//static const vmath::vec3 Z(0.0f, 0.0f, 1.0f);

	glEnable(GL_CULL_FACE);
	glDisable(GL_DEPTH_TEST);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(glb->render_prog);

	//Projection matrix
    mat4 model_matrix(translate(0.0f, 0.0f, -5.0f)*rotate(glb->y_angle, X)*rotate(glb->x_angle,Y));
	vmath::mat4 projection_matrix(vmath::frustum(-1.0f, 1.0f, -glb->aspect, glb->aspect,1.0f, 500.0f));
	glUniformMatrix4fv(glb->render_model_matrix_loc,1,GL_FALSE,model_matrix);
	glUniformMatrix4fv(glb->render_projection_matrix_loc,1,GL_FALSE,projection_matrix);
	
	glBindVertexArray(glb->vao[0]);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, glb->ebo[0]);

#if USE_PRIMITIVE_RESTART
	glEnable(GL_PRIMITIVE_RESTART);
	glPrimitiveRestartIndex(0xFFFF);
	glDrawElements(GL_TRIANGLE_STRIP, 17, GL_UNSIGNED_SHORT,NULL);
#else
	glDrawElements(GL_TRIANGLE_STRIP,8,GL_UNSIGNED_SHORT, NULL);
	glDrawElements(GL_TRIANGLE_STRIP,8,GL_UNSIGNED_SHORT, BUFFER_OFFSET(9*sizeof(GLushort)));
#endif

	glFlush();
	glutPostRedisplay();
	glutSwapBuffers();
}
void finalize(void)
{
	glUseProgram(0);
	glDeleteProgram(glb->render_prog);
	glDeleteVertexArrays(1,glb->vao);
	glDeleteBuffers(1,glb->vbo);
	delete glb;
}
void reshape(int width, int height)
{
	glViewport(0, 0, width, height);

	glb->aspect = float(width)/ float(height);
}
void keyboard(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 27:
		exit(0);
		break;
	default:
		break;
	}
}

void mouse(int button, int state, int x, int y)
{
	if(GLUT_DOWN == state){
		glb->press_x = x; glb->press_y = y;
		if(button == GLUT_LEFT_BUTTON)
			glb->mode = rotation;
	}
	else if(GLUT_UP == state)
		glb->mode = notransform;
}
void motion(int x, int y)
{
	if(glb->mode == rotation){
		glb->x_angle += (x - glb->press_x)/5.0f;
		if(glb->x_angle >180.0)
			glb->x_angle -= 360.0f;
		else if(glb->x_angle < -180)
			glb->x_angle += 360.0f;

		glb->press_x = x;

		glb->y_angle += (y - glb->press_y)/5.0f;
		if(glb->y_angle >180.0)
			glb->y_angle -= 360.0f;
		else if(glb->y_angle < -180)
			glb->y_angle += 360.0f;

		glb->press_y = y;
	}
}

int main(int argc, char** argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowPosition(140,140);
	glutInitWindowSize(1024,768);
	glutInitContextVersion(3,3);
	glutInitContextProfile(GLUT_CORE_PROFILE);
	glutCreateWindow(argv[0]);
	glewExperimental = GL_TRUE;
	if(glewInit()){
		cerr<<"Unable to initialize GLEW ... EXITING"<<endl;
		exit(EXIT_FAILURE);
	}
	init();
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutReshapeFunc(reshape);
	glutCloseFunc(finalize);
	glutMainLoop();

	return 0;

}