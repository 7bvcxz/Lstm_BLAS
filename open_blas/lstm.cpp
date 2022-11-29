#include <iostream>
#include <cstring>
#include <stdlib.h>
#include <chrono>
#include <math.h>
#include <cblas.h>
#include <x86intrin.h>

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::milliseconds ms;
typedef std::chrono::duration<float> fsec;

#define ABS(x) ((x < 1) ? (x*(-1)) : (x))
#define SIGMOID(x) (1.0f / (1.0f + exp((-1) * x)))
#define TANH(x) (1.0f / (1.0f + exp((-2) * x)) - 1)

double gemm_time = 0, gemv_time = 0, etc_time = 0;

void Compute_CPU(float* ans, float* in, float* hh_, float* ih_, float* h, float* c, float* w_hh, float* w_ih,
	             float* b, int w, int m, int n) {
	for (int k=0; k<w; k++) {
		for (int i=0; i<n; i++) {
			for (int j=0; j<m; j++) {
				ans[k*n+i] = ans[k*n+i] + in[k*m+j] * w_ih[i*m+j];
			}
		}
	}

	/*
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			ans[i] = ans[i] + w_ih[i*m+j] * in[0*m+j];
			ans[i] = ans[i] + w_hh[i*m+j] * h[0*n+j];
		}
		ans[i] = ans[i] + b_ih[i];
		ans[i] = ans[i] + b_hh[i];
	}
	*/
}

double Get_Error(float* ans, float* out, int w, int m, int n) {
	double error = 0;
	for (int i = 0; i < n; i++) {
		double tmp = ans[i] - out[i];
		error = error + ABS(tmp); 
	  	// std::cout << tmp << std::endl;
	}
	//for (int i=0; i<n; i++)
	//	std::cout << ABS(ans[i] - out[i]) << std::endl;
	return error;
}

void LSTM_Cell(int k, float* w_hh, float* h, float* hh_, float* ih_, float* b, float* c, int w, int m, int n) {
	int incx=1, incy=1, alpha=1, beta=0, lda=m;
	
	auto start0 = Time::now();
	cblas_sgemv(CblasRowMajor, CblasNoTrans, n, m, alpha, w_hh, lda, &h[k*n], incx, beta, hh_, incy);
	auto end0 = Time::now();
	fsec time0 = end0 - start0;
	gemv_time = gemv_time + time0.count();
	
	auto start1 = Time::now();
	for (int i=0; i<n; i++)
		hh_[i] = hh_[i] + ih_[i] + b[i];
	
	int Lchunk = n/4;
	float* chk_0 = hh_;
	float* chk_1 = hh_ + Lchunk;
	float* chk_2 = hh_ + 2*Lchunk;
	float* chk_3 = hh_ + 3*Lchunk;

	float* c_old = c + w * k;
	float* c_new = c + w * (k+1);
	float* h_new = h + w * (k+1);

	float *in_gate = (float *)calloc(Lchunk, sizeof(float));	  // [n/4]
	float *forget_gate = (float *)calloc(Lchunk, sizeof(float));  // [n/4]
	float *cell_gate = (float *)calloc(Lchunk, sizeof(float));	  // [n/4]
	float *out_gate = (float *)calloc(Lchunk, sizeof(float));	  // [n/4]

	for (int i=0; i<Lchunk; i++) {
		in_gate[i] = SIGMOID(chk_0[i]);
		forget_gate[i] = SIGMOID(chk_1[i]);
		cell_gate[i] = TANH(chk_2[i]);
		out_gate[i] = SIGMOID(chk_3[i]);

		c_new[i] = (forget_gate[i] * c_old[i]) + in_gate[i] * cell_gate[i];
		h_new[i] = out_gate[i] * TANH(c_new[i]);
	}
	auto end1 = Time::now();
	fsec time1 = end1 - start1;
	etc_time = etc_time + time1.count();
}

int main(int argc, char **argv) {
	int w = 2;  // num_word
	int m = 1024; // num_col
	int n = 4096; // num_row

	/* Initialize */
	float *in = (float *)calloc(w * m, sizeof(float));     // [w, m]
	float *out = (float *)calloc(w * n, sizeof(float));    // [w, n]
	float *ans = (float *)calloc(w * n, sizeof(float));    // [w, n]
	float *hh_ = (float *)calloc(n, sizeof(float));        // [n]
	float *ih_ = (float *)calloc(w * n, sizeof(float));	   // [w, n]
	float *h = (float *)malloc(sizeof(float) * w * n);     // [w, n]
	float *c = (float *)malloc(sizeof(float) * w * n);     // [w, n]
	float *w_hh = (float *)malloc(sizeof(float) * n * m);  // [n, m]
	float *w_ih = (float *)malloc(sizeof(float) * n * m);  // [n, m]
	float *b = (float *)malloc(sizeof(float) * n);		   // [n]

	  // Input //
	for (int k = 0; k < w; k++)
		for (int j = 0; j < m; j++)
			in[k * m + j] = float(RAND_MAX) / (rand()*100);

	  // LSTM hidden, cell state // 
	for (int j = 0; j < n; j++) {
		h[0 * n + j] = 0;
		c[0 * n + j] = 0;
	}

	  // Weight //
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			w_hh[i * m + j] = float(RAND_MAX) / (rand()*100);
			w_ih[i * m + j] = float(RAND_MAX) / (rand()*100);
		}
	}

	  // Bias //
	for (int i = 0; i < n; i++)
		b[i] = float(RAND_MAX) / (rand()*100);

	  // Cache Flush W_HH
	for (int i=0; i<n; i++) {
		for (int j=0; j<m; j++) {
			_mm_clflush(&w_hh[i * m + j]);
		}
	}

	auto start = Time::now();

	/* Calculate Custom LSTM Answer */
	  // First GEMM (Input - Hidden)
	int incx=1, incy=1, alpha=1, beta=0, lda=m, ldb=n, ldc=n;
	//cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, w, n, m, alpha, in, lda, w_ih, ldb, beta, ih_, ldc);

	auto end = Time::now();
	fsec time = end - start;
	gemm_time = time.count();

	  // Second GEMV (Hidden - Hidden)
	for (int k=0; k<(w-1); k++)
		LSTM_Cell(k, w_hh, h, hh_, ih_, b, c, w, m, n);

	//for (int k=0; k<w; k++)
	//	for (int i=0; i<n; i++)
	//		out[k*n+i] = ih_[k*n+i];


	std::cout << "tot_time : " << gemm_time + gemv_time + etc_time << "s\n";
	std::cout << "  gemm_time : " << gemm_time << "s\n";
	std::cout << "  gemv_time : " << gemv_time << "s\n";
	std::cout << "  etc_time  : " << etc_time << "s\n";
	
	/* Calculate CPU Answer */
	//Compute_CPU(ans, in, hh_, ih_, h, c, w_hh, w_ih, b, w, m, n);

	/* Calculate ERROR */
	//double error = Get_Error(ans, out, w, m, n);
	//std::cout << "ERROR: " << error << std::endl;
	
	return 0;
}
