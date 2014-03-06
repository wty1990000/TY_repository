#include "vgl.h"
#include <vmath.h>

#include "LoadShaders.h"

#include <stdio.h>
#include <iostream>

using namespace std;
using namespace vmath;


struct global{
	float aspect;
	GLuint render_prog;
	GLuint vao[1];
	GLuint vbo[1];
	GLuint ebo[1];

	GLint render_model_matrix_loc;
	GLint render_projection_matrix_loc;
}*glb;



void init()
{
	glb = new global();

    static ShaderInfo shader_info[] =
    {
        { GL_VERTEX_SHADER, "vs.glsl" },
        { GL_FRAGMENT_SHADER, "fs.glsl" },
        { GL_NONE, NULL }
    };

    glb->render_prog = LoadShaders(shader_info);

    glUseProgram(glb->render_prog);

    // "model_matrix" is actually an array of 4 matrices
    glb->render_model_matrix_loc = glGetUniformLocation(glb->render_prog, "model_matrix");
    glb->render_projection_matrix_loc = glGetUniformLocation(glb->render_prog, "projection_matrix");

    // A single triangle
    static const GLfloat vertex_positions[] =
    {
        -1.0f, -1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  0.0f, 1.0f,
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 1.0f,
    };

    // Color for each vertex
    static const GLfloat vertex_colors[] =
    {
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 0.0f, 1.0f,
        1.0f, 0.0f, 1.0f, 1.0f,
        0.0f, 1.0f, 1.0f, 1.0f
    };

    // Indices for the triangle strips
    static const GLushort vertex_indices[] =
    {
        0, 1, 2
    };

    // Set up the element array buffer
    glGenBuffers(1, glb->ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, glb->ebo[0]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(vertex_indices), vertex_indices, GL_STATIC_DRAW);

    // Set up the vertex attributes
    glGenVertexArrays(1, glb->vao);
    glBindVertexArray(glb->vao[0]);

    glGenBuffers(1, glb->vbo);
    glBindBuffer(GL_ARRAY_BUFFER, glb->vbo[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertex_positions) + sizeof(vertex_colors), NULL, GL_STATIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertex_positions), vertex_positions);
    glBufferSubData(GL_ARRAY_BUFFER, sizeof(vertex_positions), sizeof(vertex_colors), vertex_colors);

    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(0));
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(sizeof(vertex_positions)));
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
}
void display()
{
    mat4 model_matrix;

    // Setup
    glEnable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Activate simple shading program
    glUseProgram(glb->render_prog);

    // Set up the model and projection matrix
    vmath::mat4 projection_matrix(vmath::frustum(-1.0f, 1.0f, -glb->aspect, glb->aspect, 1.0f, 500.0f));
    glUniformMatrix4fv(glb->render_projection_matrix_loc, 1, GL_FALSE, projection_matrix);

    // Set up for a glDrawElements call
    glBindVertexArray(glb->vao[0]);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, glb->ebo[0]);

    // Draw Arrays...
    model_matrix = vmath::translate(-3.0f, 0.0f, -5.0f);
    glUniformMatrix4fv(glb->render_model_matrix_loc, 1, GL_FALSE, model_matrix);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    // DrawElements
    model_matrix = vmath::translate(-1.0f, 0.0f, -5.0f);
    glUniformMatrix4fv(glb->render_model_matrix_loc, 1, GL_FALSE, model_matrix);
    glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_SHORT, NULL);

    // DrawElementsBaseVertex
    model_matrix = vmath::translate(1.0f, 0.0f, -5.0f);
    glUniformMatrix4fv(glb->render_model_matrix_loc, 1, GL_FALSE, model_matrix);
    glDrawElementsBaseVertex(GL_TRIANGLES, 3, GL_UNSIGNED_SHORT, NULL, 1);

    // DrawArraysInstanced
    model_matrix = vmath::translate(3.0f, 0.0f, -5.0f);
    glUniformMatrix4fv(glb->render_model_matrix_loc, 1, GL_FALSE, model_matrix);
    glDrawArraysInstanced(GL_TRIANGLES, 0, 3, 1);

	glFlush();
	glutSwapBuffers();
}
void Finalize(void)
{
    glUseProgram(0);
    glDeleteProgram(glb->render_prog);
    glDeleteVertexArrays(1, glb->vao);
    glDeleteBuffers(1, glb->vbo);
}
void Reshape(int width, int height)
{
    glViewport(0, 0 , width, height);

    glb->aspect = float(height) / float(width);
}
void keyboard(unsigned char key, int x, int y)
{
	switch(key)
	{
		case 27:
			exit(0);
			break;
	}
		
			
}


int main(int argc, char** argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowPosition(140,140);
	glutInitWindowSize(1024, 768);
	glutInitContextVersion(3, 3);
	glutInitContextProfile(GLUT_CORE_PROFILE);
	glutCreateWindow(argv[0]);
	glewExperimental = GL_TRUE;  //must be added to validate glGenVetexArray
	if (glewInit()) {
		cerr << "Unable to initialize GLEW ... exiting" << endl;
		exit(EXIT_FAILURE);
	}
	init();
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutReshapeFunc(Reshape);
	glutCloseFunc(Finalize);
	glutMainLoop();
}