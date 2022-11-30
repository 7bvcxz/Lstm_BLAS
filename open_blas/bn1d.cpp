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

double tot_time = 0;

void BN1d(int w, int f, float* in, float* sig, float* mu, float* out) {
	for (int i=0; i<w; i++) {
		for (int j=0; j<f; j++) {
			out[i*f + j] = in[i*f + j] * sig[j] + mu[j];
		}
	}
}

int main(int argc, char **argv) {
	int w = 576;  // num_word
	int f = 1024; // num_feature

	/* Initialize */
	float *in = (float *)calloc(w * f, sizeof(float));	// [w, f]
	float *out = (float *)calloc(w * f, sizeof(float));	// [w, f]
	float *ans = (float *)calloc(w * f, sizeof(float));	// [w, f]
	float *sig = (float *)malloc(sizeof(float) * f);	// [f]
	float *mu = (float *)malloc(sizeof(float) * f);		// [f]

	  // Input //
	for (int i=0; i<w; i++)
		for (int j=0; j<f; j++) 
			in[i*f + j] = float(RAND_MAX) / (rand()*100);

	  // Sigma, Mu //
	for (int i=0; i<f; i++) {
		sig[i] = float(RAND_MAX) / (rand()*100);
		mu[i] = float(RAND_MAX) / (rand()*100);
	}
	
	/* Flush Cache */
	  // Cache Flush by simple Big data
	float *flush0 = (float *)malloc(128 * 1024 * 1024);  // size : 16MB
	float *flush1 = (float *)malloc(128 * 1024 * 1024);  // size : 16MB
	int flush_m = 128 * 1024 * 1024 / sizeof(float);

	for (int i=0; i<flush_m; i++)
		flush1[i] = flush0[i] + flush1[i];

	auto start = Time::now();

	/* Calculate Custom BN1d Answer */
	BN1d(w, f, in, sig, mu, out);

	auto end = Time::now();
	fsec time = end - start;
	std::cout << "tot_time : " << time.count() << "s\n";
	
	/* Calculate CPU Answer */
	//Compute_CPU(ans, in, hh_, ih_, h, c, w_hh, w_ih, b, w, m, n);

	/* Calculate ERROR */
	//double error = Get_Error(ans, out, w, m, n);
	//std::cout << "ERROR: " << error << std::endl;
	
	return 0;
}
