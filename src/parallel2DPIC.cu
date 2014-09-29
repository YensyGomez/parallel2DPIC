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

float L,LL; int N, C,itera;

using namespace std;


__device__ float distribution (float vb, float aleatorio, curandState *states)     //generador de distribución maxwelliana para la velocidad
{

  // Genera un valor random v
   float fmax = 0.5 * (1.0 + exp (-2.0 * vb * vb));
   float vmin = - 5.0 * vb;
   float vmax = + 5.0 * vb;
   float v;
   float f;
   float x;
   int Idx = blockIdx.x*blockDim.x + threadIdx.x;

   while(true){
	   v = vmin + ((vmax - vmin) * aleatorio);
	   f = 0.5 * (exp (-(v - vb) * (v - vb) / 2.0) +
			    exp (-(v + vb) * (v + vb) / 2.0));
	   x = fmax * aleatorio;
	   if(x > f) aleatorio = curand_uniform(states + Idx);
	   else return v;
   }

}

__global__ void distribucionParticulas(float *rx,float *ry,float *vx,float *vy,int N,curandState *states,float vb,float L){
	int Idx = blockIdx.x*blockDim.x + threadIdx.x;

	unsigned int seed = (unsigned int) (clock() * Idx);
	curand_init(seed, 0, 0, states + Idx);

	if(Idx < N){
		 rx[Idx] = L*curand_uniform(states + Idx);    //inicializando la posicion aleatoria en x
		 ry[Idx] = L*curand_uniform(states + Idx);
		 vx[Idx] = distribution(vb,curand_uniform(states + Idx),states);//;L*curand_uniform_float(states + Idx);//distribution(vb,states);                          //inicializa la velocidad con una distribucion maxwelliana
		 vy[Idx] = distribution(vb,curand_uniform(states + Idx),states);//L*curand_uniform_float(states + Idx);//distribution(vb,states);                          //inicializa la velocidad con una distribucion maxwelliana

	}

}


int main(){
	// Parametros
	L = 25.0;            // dominio de la solucion 0 <= x <= L (en longitudes de debye)
	//L=LL*LL;
	N = 10000;            // Numero de particulas
	C = 50;          // Numero de celdas EN UNA DIMENSION, EL TOTAL DE CELDAS ES C*C
	float vb = 3.0;    // velocidad rayo promedio
	//float dt=0.1;    // delta tiempo (en frecuencias inversas del plasma)
	//float tmax=10000;  // cantidad de iteraciones. deben ser 100 mil segun el material
	//int skip = int (tmax / dt) / 10; //saltos del algoritmo para reportar datos
	//int itera=0;
	float *rx_h,*ry_h,*vx_h,*vy_h;
	float *rx_d,*ry_d,*vx_d,*vy_d;

	int size = N*sizeof(float);
	//reserva en memoria al host
	rx_h = (float *)malloc(size);
	ry_h = (float *)malloc(size);
	vx_h = (float *)malloc(size);
	vy_h = (float *)malloc(size);
	//reserva de memoria del dispositivo.
	cudaMalloc((void **)&rx_d,size);
	cudaMalloc((void **)&ry_d,size);
	cudaMalloc((void **)&vx_d,size);
	cudaMalloc((void **)&vy_d,size);
	//valores aleatorios.
	curandState *devStates;
	cudaMalloc((void **) &devStates, N * sizeof(curandState));

	//lanzar el kernel. El primer parámetro que va al llamarse un kernel es la cantidad de hilos que queremos que haya en cada bloque.
	//despues va la cantidad de bloques

	float blockSize = 1024;
	dim3 dimBlock (ceil(N/blockSize), 1, 1);
	dim3 dimGrid (blockSize, 1, 1);


	distribucionParticulas<<<blockSize,dimBlock>>>(rx_d,ry_d,vx_d,vy_d,N,devStates,vb,L);
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
	for(int i = 0; i < N ;i++){
		printf("%f %f %f %f\n",rx_h[i],ry_h[i],vx_h[i],vy_h[i]);
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
