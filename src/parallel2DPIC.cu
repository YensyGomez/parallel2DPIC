/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include <fstream>
#include <vector>
#include <fstream>
#include <curand_kernel.h>

double L,LL; int N, C,itera;

using namespace std;


__device__ double distribution (double vb,curandState *states)     //generador de distribuci�n maxwelliana para la velocidad
{
  // inicializa el generador aleatorio
  int flag = 0;
  if (flag == 0)
    {
	  unsigned int seed = (unsigned int) (clock());
	  curand_init(seed, 0, 0, states);
      flag = 1;
    }

  // Genera un valor random v
   double fmax = 0.5 * (1. + exp (-2. * vb * vb));
   double vmin = - 5. * vb;
   double vmax = + 5. * vb;
   double v = vmin + (vmax - vmin) * double(curand_uniform_double(states)) / double (RAND_MAX);



  // Acceptar y reinyectar particulas
  double f = 0.5 * (exp (-(v - vb) * (v - vb) / 2.) +
		    exp (-(v + vb) * (v + vb) / 2.));
  double x = fmax *double(curand_uniform_double(states))/ double (RAND_MAX);
  if (x > f) return distribution (vb,states);
  else
  {
	  return v;

  }

}

__global__ void distribucionParticulas(double *rx,double *ry,double *vx,double *vy,int N,curandState *states,double vb,double L){
	int Idx=threadIdx.x+blockDim.x*threadIdx.x;
	unsigned int seed = (unsigned int) (clock() * Idx);
	curand_init(seed, 0, 0, states + Idx);
	if(Idx<N){
		 rx[Idx]=(L*double (curand_uniform_double(states + Idx)) / double (RAND_MAX));    //inicializando la posicion aleatoria en x
		 ry[Idx]=(L*double (curand_uniform_double(states + Idx)) / double (RAND_MAX));
		 vx[Idx]=(distribution(vb,states));                          //inicializa la velocidad con una distribucion maxwelliana
		 vy[Idx]=(distribution(vb,states));                          //inicializa la velocidad con una distribucion maxwelliana

	}

}


int main(){
	// Parametros
	  L =250000000.0;            // dominio de la solucion 0 <= x <= L (en longitudes de debye)
	  //L=LL*LL;
	  N =10000;            // Numero de particulas
	  C = 50;            // Numero de celdas EN UNA DIMENSION, EL TOTAL DE CELDAS ES C*C
	  double vb = 3.0;    // velocidad rayo promedio
	  //double dt=0.1;    // delta tiempo (en frecuencias inversas del plasma)
	  //double tmax=10000;  // cantidad de iteraciones. deben ser 100 mil segun el material
	  //int skip = int (tmax / dt) / 10; //saltos del algoritmo para reportar datos
	  //int itera=0;
	  double *rx_h,*ry_h,*vx_h,*vy_h;
	  double *rx_d,*ry_d,*vx_d,*vy_d;

	  int size= N*sizeof(double);
	  //reserva en memoria al host
	  rx_h= (double *)malloc(size);
	  ry_h= (double *)malloc(size);
	  vx_h= (double *)malloc(size);
	  vy_h= (double *)malloc(size);
	  //reserva de memoria del dispositivo.
	  cudaMalloc(&rx_d,size);
	  cudaMalloc(&ry_d,size);
	  cudaMalloc(&vx_d,size);
	  cudaMalloc(&vy_d,size);
	  //valores aleatorios.
	  curandState *devStates;
	  cudaMalloc((void **) &devStates, N * sizeof(curandState));

	//lanzar el kernel.
	  distribucionParticulas<<<ceil(N/1024),1024>>>(rx_d,ry_d,vx_d,vy_d,N,devStates,vb,L);
	// ontener los resultados.
	//posición en x.
	  cudaMemcpy(rx_h, rx_d, size, cudaMemcpyDeviceToHost);

	// posición en y.
	  cudaMemcpy(ry_h, ry_d, size, cudaMemcpyDeviceToHost);

	// velocidad en x.
	  cudaMemcpy(vx_h, vx_d, size, cudaMemcpyDeviceToHost);

	//velocidad en y.
	  cudaMemcpy(vy_h, vy_d, size, cudaMemcpyDeviceToHost);

	//Imprimir el resultado
	     double resultado1 =0;
	     double resultado2 =0;
	     double resultado3 =0;
	     double resultado4 =0;
	     for(int i=0;i<N;i++){
	    	  resultado1 =rx_h[i];
	    	  resultado2 =ry_h[i];
	    	  resultado3 =vx_h[i];
	    	  resultado4 =vy_h[i];
	    	 printf("%f %f %f %f\n",resultado1,resultado2,resultado3,resultado4);
	     }

	  free(rx_h);
	  free(ry_h);
	  free(vx_h);
	  free(vy_h);
	  cudaFree(rx_d);
	  cudaFree(ry_d);
	  cudaFree(vx_d);
	  cudaFree(vy_d);

	return (0);

}
