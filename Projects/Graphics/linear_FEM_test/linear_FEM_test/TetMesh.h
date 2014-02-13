#ifndef _TETMESH_H_
#define _TETMESH_H_

#include <glm/glm.hpp>
#include "utilities.h"

/* --------- Tetrahedral data structure -----------*/
struct Tetrahedron{
	int iIndex[4];			//hold vertex index
	float fVolume;			//hold the volume
	float fPlastic[6];		//hold plastic value
	glm::vec3 e1,e2,e3;		//edges;
	glm::mat3 Re;			//Rotational warp
	glm::mat3 K4[4][4];		//Stiffness matrix for each element
	glm::vec3 B[4];			//Jacobian of shapefunction
};

static std::vector<Tetrahedron> tetrahedra;


#endif // !_TETMESH_H_


