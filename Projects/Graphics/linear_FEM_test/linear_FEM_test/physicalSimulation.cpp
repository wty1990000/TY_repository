#include "physicalSimulation.h"
#include <limits>
using namespace std;

//Initialize the physical parameters
void initializePhysics()
{
	physicalglobals = new PhysicalGlobalVariables();
}
//compute the volume of a tetraheron
float GetTetrahedronVolume(glm::vec3 e1, glm::vec3 e2, glm::vec3 e3)
{
	return (glm::dot(e1, glm::cross(e2,e3)))/6.0f;
}
//Add one new tetraheron to the existing tetmesh vector
void addTetraheron(int i0, int i1, int i2, int i3)
{
	Tetrahedron t;
	t.iIndex[0] = i0;
	t.iIndex[1] = i1;
	t.iIndex[2] = i2;
	t.iIndex[3] = i3;

	tetrahedra.push_back(t);
}
//Generate the initial mesh
void genMesh(size_t xdim, size_t ydim, size_t zdim, float fWidth, float fHeight, float fDepth)
{
	physicalglobals->total_points = (xdim+1)*(ydim+1)*(zdim+1);
	physicalglobals->P.resize(physicalglobals->total_points);
	physicalglobals->Xi.resize(physicalglobals->total_points);
	physicalglobals->IsFixed.resize(physicalglobals->total_points);
	int ind = 0;
	
	//build every tetrohedron's vertex
	for(unsigned int  x=0; x<=xdim; x++){
		for(unsigned int y=0; y<=ydim; y++){
			for(unsigned int z=0; z<=zdim; z++){
				physicalglobals->Xi[ind] = physicalglobals->P[ind] 
				= glm::vec3(fWidth*x,fHeight*z,fDepth*y);
				if(physicalglobals->Xi[ind].x < 0.01)
					physicalglobals->IsFixed[ind] = true;

				ind++;
			}
		}
	}
	for(size_t i=0; i<physicalglobals->total_points; i++){
		physicalglobals->P[i].y += 0.5;
		physicalglobals->P[i].z -= zdim/2 * fDepth;
	}
	for (size_t i = 0; i < xdim; ++i) {
		for (size_t j = 0; j < ydim; ++j) {
			for (size_t k = 0; k < zdim; ++k) {
				int p0 = (i * (ydim + 1) + j) * (zdim + 1) + k;
				int p1 = p0 + 1;
				int p3 = ((i + 1) * (ydim + 1) + j) * (zdim + 1) + k;
				int p2 = p3 + 1;
				int p7 = ((i + 1) * (ydim + 1) + (j + 1)) * (zdim + 1) + k;
				int p6 = p7 + 1;
				int p4 = (i * (ydim + 1) + (j + 1)) * (zdim + 1) + k;
				int p5 = p4 + 1;
				// Ensure that neighboring tetras are sharing faces
				if ((i + j + k) % 2 == 1) {
					addTetraheron(p1,p2,p6,p3);
					addTetraheron(p3,p6,p4,p7);
					addTetraheron(p1,p4,p6,p5);
					addTetraheron(p1,p3,p4,p0);
					addTetraheron(p1,p6,p4,p3); 
				} else {
					addTetraheron(p2,p0,p5,p1);
					addTetraheron(p2,p7,p0,p3);
					addTetraheron(p2,p5,p7,p6);
					addTetraheron(p0,p7,p5,p4);
					addTetraheron(p2,p0,p7,p5); 
				}
				physicalglobals->iNumofTetrohedron+=5;
			}
		}
	}
}
//Compute vetex force
void computeforce()
{
	size_t i=0;
	for(i = 0; i< physicalglobals->total_points; i++){
		physicalglobals->F[i] = glm::vec3(0);

		//Only consider the gravity for non-fixed points
		physicalglobals->F[i] += gravAcceleration*physicalglobals->MASS[i];
	}
}
//Reassemble the lumped mass matrix
void recalcmassmatrix()
{
	for(size_t i=0; i< physicalglobals->total_points;i++){
		if(physicalglobals->IsFixed[i])
			physicalglobals->MASS[i] = numeric_limits<float>::max( );
		else
			physicalglobals->MASS[i] = 1.0f/physicalglobals->total_points;
	}
	for(int i =0; i < physicalglobals->iNumofTetrohedron; i++){
		float fM = (fDensity * tetrahedra[i].fVolume)*0.25f;
		physicalglobals->MASS[tetrahedra[i].iIndex[0]] += fM;
		physicalglobals->MASS[tetrahedra[i].iIndex[1]] += fM;
		physicalglobals->MASS[tetrahedra[i].iIndex[2]] += fM;
		physicalglobals->MASS[tetrahedra[i].iIndex[3]] += fM;
	}
}
//Calculate the nodal forces
void computeforce()
{
	for (int i = 0; i < tetrahedra.size(); i++){
		for (int j=0; j<physicalglobals->total_points; j++){
			double temp = 0.0;
			for( int k =0; k< physicalglobals->total_points;k++)
			{
				if(j>=k){
					temp += physicalglobals->K_row[k]*physicalglobals->A_row[j];
					
					if(j>k)
						
				}
			}
		}
	}

}