#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <omp.h>
double prediction(double *X, double *W, double B, int dim)
{
  double sum = 0.0;
  //calculate f(x_i) = W_1*X_1 + W_2*X_2 .... W_i*X_i + B
  for(int i = 0; i < dim; i++)
  {
    sum += X[i] * W[i];
  }
  sum += B;
  return sum;
}
//caclulate derivate of Mean squared error loss function with respect to W_i
double derivative(int N, int M, double *W, double B, int w_i, double X[N][M], double Y[N]) {
  double dldw = 0.0;
  for(int i = 0; i < N; i++) {
    //extract row i from X
    double row[M];
    for(int j = 0; j < M; j++) { 
      row[j] = X[i][j];
    }
    double pred = prediction(row, W, B, M);
    dldw += -2 * row[w_i] * (Y[i] - pred);
  }
  return dldw/N;
}
int main(int argc, char **argv)
{
  srand(time(NULL));
  //double start = omp_get_wtime();
  const double LEARNING_RATE = 0.001;  
  //inititalize global variables, n(size of training data), m(number of features), X(feature data array [n X m]), Y(target data array[N x 1], W(weights for linear regression equation)
  int N = atoi(argv[1]);
  int M = atoi(argv[2]);
  int iterations = atoi(argv[3]);
  //int threads = atoi(argv[4]); //number of threads
  //omp_set_num_threads(threads);
  double Y[N];
  double X[N][M];
  double W[M];
  for(int i = 0; i < M; i++)
  {
    W[i] = 0.0;
  }
  double B = 0.0;
  double dldw_i;
  //intialize training data
  for(int i = 0; i < N; i++)
  {
    Y[i] = 0.0;
    for(int j = 0; j < M; j++)
    {
        X[i][j] = ((double) rand())/((double) RAND_MAX);
        //printf("training data point = %e\n", trainingData[i][j]);
	Y[i] += X[i][j];
    }
    //Y[i] = ((double) rand())/((double) RAND_MAX);
    //printf("target = %e\n", Y[i]);
  }
  //run gradient descent for x iterations
#pragma omp parallel for private(dldw_i)
    for(int i = 0; i < iterations; i++) {
    //double pred = prediction(X, W, B, M);
    for(int j = 0; j < M; j++) {
            dldw_i = derivative(N, M, W, B, j, X, Y);
            //update Weight with derivative
            W[j] = W[j] - LEARNING_RATE * dldw_i;
    }
  }
  for(int i = 0; i < M; i++) {
    printf("Weight %i is %e\n", i, W[i]);
  }

  //printf("%e\n", omp_get_wtime() - start);
  return 0;
}
