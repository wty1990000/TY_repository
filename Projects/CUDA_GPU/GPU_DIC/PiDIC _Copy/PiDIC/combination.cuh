#ifndef _COMBINATION_H_
#define _COMBINATION_H_

void combined_functions(const double* h_InputIMGR, const double* h_InputIMGT, int m_iWidth, int m_iHeight, 
						double* m_dZNCC,
						double* m_iU, double *m_iV,
						float& precompute_tme, float& fft_time, float& icgn_time, float& total_time);

#endif // !_COMBINATION_H_
