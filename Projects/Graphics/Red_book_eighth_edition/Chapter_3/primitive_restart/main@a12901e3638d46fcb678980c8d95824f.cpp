#include "vgl.h"
#include <vmath.h>

#include "LoadShaders.h"
#include <iostream>

#define USE_PRIMITIVE_RESTART 1

using namespace std;
using namespace vmath;

typedef struct global_t{
	float aspect;
	GLuint render_prog;
	GLuint vao[1];
	GLuint vbo[1];
	GLuint ebo[1];

	GLint render_model_matrix_loc;
	GLint render_projection_matrix_loc;
}global;

global *glb;

void init()
{
	glb = new global();

	static ShaderInfo shader_info[] = 
	{
		{GL_VERTEX_SHADER, "vl.glsl"},
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

	glGenBuffers(GL_ARRAY_BUFFER, glb->vbo);
	glBindBuffer(GL_ARRAY_BUFFER, glb->vbo[0]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(cube_positions) + sizeof(cube_color), NULL, GL_STATIC_DRAW);
	glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(cube_positions), cube_positions);
	glBufferSubData(GL_ARRAY_BUFFER,sizeof(cube_positions), sizeof(cube_color),cube_color);
	
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(0));
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(sizeof(cube_positions)));
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
}
void display()
{
	static float t = float(GetTickCount() & 0x1FFF) / float(0x1FFF);
	static const vmath::vec3 X(1.0f, 0.0f, 0.0f);
	static const vmath::vec3 Y(0.0f, 1.0f, 0.0f);
	static const vmath::vec3 Z(0.0f, 0.0f, 1.0f);

	glEnable(GL_CULL_FACE);
	glDisable(GL_DEPTH_TEST);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(glb->render_prog);

	//Projection matrix
	vmath::mat4 model_matrix(translate(0.0f, 0.0f, -5.0f)*rotate(t * 360.0f, Y)*rotate(t*720,Z));
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
	glDrawElements(GL_TRIANGLE_STRIP,8,GL_UNSIGNED_SHROT, BUFFER_OFFSET(9*sizeof(GLushort)));
#endif

	glFlush();
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

int main(int argc, char** argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowPosition(140,140);
	glutInitWindowSize(1024,768);
	glutInitContextVersion(3,3);
	glutInitContextProfile(GLUT_CORE_PROFILE);
	glutCreateWindow("Primitive Restart");
	glewExperimental = GL_TRUE;
	if(glewInit()){
		cerr<<"Unable to initialize GLEW ... EXITING"<<endl;
		exit(EXIT_FAILURE);
	}
	init();
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutReshapeFunc(reshape);
	glutCloseFunc(finalize);
	glutMainLoop();

	return 0;

}