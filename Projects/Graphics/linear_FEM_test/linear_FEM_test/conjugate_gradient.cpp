#include "conjugate_gradient.h"
#include "physicalSimulation.h"


void initialize_CGsolver()
{
	cgglobals = new ConjugateGradient();
}
//Conjugate gradient solver
void conjugate_gradient_solver(const float &dt, ConjugateGradient &CG)
{
	for(size_t k=0; k < physicalglobals->total_points; k++){
		if(physicalglobals->IsFixed[k])
			continue;
		cgglobals->residual[k] = physicalglobals->b[k];

		matrix_iterator Abegin = physicalglobals->A_row[k].begin();
		matrix_iterator Aend = physicalglobals->A_row[k].end();
		for(matrix_iterator it_A = Abegin; it_A != Aend; it_A++){
			
		}

	}
}