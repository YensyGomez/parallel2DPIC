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
#include <cufft.h>

float L,LL;

int N, C,itera;

using namespace std;

// función Maxwelliana de la distribución de las partículas.
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

 return 0;
}
//Distribución aleatoria de las partículas.
__global__ void distribucionParticulas(float *rx,float *ry,float *vx,float *vy,int N,curandState *states,float vb,float L, int seed){
	int Idx = blockIdx.x*blockDim.x + threadIdx.x;

	seed = (unsigned int) (clock() * Idx);
	curand_init(seed, 0, 0, states + Idx);

	if(Idx < N){
		rx[Idx] = L*curand_uniform(states + Idx);    //inicializando la posicion aleatoria en x
		ry[Idx] = L*curand_uniform(states + Idx);
		vx[Idx] = distribution(vb,curand_uniform(states + Idx),states);//;L*curand_uniform_float(states + Idx);//distribution(vb,states);                          //inicializa la velocidad con una distribucion maxwelliana
		vy[Idx] = distribution(vb,curand_uniform(states + Idx),states);//L*curand_uniform_float(states + Idx);//distribution(vb,states);                          //inicializa la velocidad con una distribucion maxwelliana

	}


}
// inicialización de la densidad.
__global__ void inicializacionDensidad(float *ne,int C){
	int Id=blockIdx.x*blockDim.x + threadIdx.x;
	if(Id<(C*C)){
		ne[Id] = 0.0;
	}
 }

__global__ void inicializacionValoresReales(float *vr,int C){
	int Id=blockIdx.x*blockDim.x + threadIdx.x;
	if(Id<(C*C)){
		vr[Id] = 0.0;
	}
 }

//Calculo de la densidad en cada celda.

__global__ void calculoDensidadInicializacionCeldas(float *rx, float *ry, int *jx,int *jy,float *yx, int N, int C,float L){
	int Id = blockIdx.x*blockDim.x + threadIdx.x;
	 float dx = L / float (C);
	 //float dxx = L /float(C*C);
	if(Id < N){
		 jx[Id] = int(rx[Id]/dx); //posicion en x de la particula
		 jy[Id] = int(ry[Id]/dx); //posicion en y de la particula
		 yx[Id] = (rx[Id]/dx) - (float)jx[Id]; //posicion exacta de la particula en x de la celda "j"
    }

}
__global__ void calculoDensidad(float *ne, int *jx, int *jy,float *yx, int C, float L, int N){
	 float dxx = L /float(C*C);
	 //int Id = blockIdx.x*blockDim.x + threadIdx.x;
	 for(int i=0; i<N; i++){
		ne[(jy[i]*C)+jx[i]] += (1. - yx[i])/dxx;
		if(jx[i]+1 == C) ne[(jy[i]*C)] += yx[i]/dxx;
		else ne[(jy[i]*C)+jx[i]+1] += yx[i]/dxx;
	 }

}

__global__ void normalizacionDensidad(float *ne,float *n, int N, int C, float L){
	 int Id = blockIdx.x*blockDim.x + threadIdx.x;
	 if(Id<C*C){
		 n[Id]=float(C*C)*ne[Id]/float(N)-1;
	 }

}


void Output(float *ne_d, float *n_d, int *jx_d,int *jy_d,float *yx_d,int C,float L,int N){

	float blockSize = 1024;
	dim3 dimBlock (ceil(N/blockSize), 1, 1);
	dim3 dimBlock2 (ceil(C*C/blockSize), 1, 1);
	dim3 dimGrid (blockSize, 1, 1);

	calculoDensidad<<<1,1>>>(ne_d,jx_d,jy_d,yx_d,C,L,N);//proceso de mejora.
	cudaDeviceSynchronize();
	normalizacionDensidad<<<blockSize,dimBlock2>>>(ne_d,n_d,N,C,L);
	cudaDeviceSynchronize();


}

//Asiganción inicial para el calculo de la transformada.

__global__ void realToComplex(float *n, cufftComplex *n_C, int C){

  int Idx = blockIdx.x*blockDim.x + threadIdx.x;
  int Idy = blockIdx.y*blockDim.y + threadIdx.y;
  int index= Idx*C+Idy;
  if(Idx<C && Idy< C){
	  n_C[index].x=n[index];
	  n_C[index].y=0.0f;

  }
}
//Normalizacion de la salida de la transformada hacia adelante
__global__ void normalizaciónSalidaUt_F(cufftComplex *n_C, cufftComplex *uT_F, int C ){
	int Idx = blockIdx.x*blockDim.x + threadIdx.x;
	int Idy = blockIdx.y*blockDim.y + threadIdx.y;
	int index= Idx*C+Idy;
	  if(Idx<C && Idy< C){
		  uT_F[index].x=n_C[index].x/float(C*C*C*C);
		  uT_F[index].y=n_C[index].y/float(C*C*C*C);

	  }

}
// calculo de Poisson.
__global__ void calculoPoisson(cufftComplex *uT_F, cufftComplex *uC_P, float L, int C){
	uT_F[0.0].x =0.;

	float i(0.0, L); //creamos una variable compleja para poder aplicar la discretizacion.
	float W = exp(2.0 * M_PI * i / float (C));
	float Wm = L, Wn = L;
	float denom = 4.0;
	float dx=L/float(C-1); //delta x, el tamaño de la malla
	for (int i= 0; i< C; i+)
	{
		for (int j= 0; j< C; j++)
		{
			denom -= Wm + L / Wm + Wn + L / Wn; //se calcula el denominador para cada celda, segun el equema de discretizacion
			if (denom != 0.0){
				uT_F[i*C+j].x *= dx *dx / denom;
				uT_F[i*C+j].y *= dx *dx / denom;
			}
			Wn *= W;//se multiplica por la constante W
		}
		Wm *= W;
	}

	uT_F[0.0].x=0.;
	uT_F[0.0].y=0.;

	// asiganr a uC_P
}
__global__ void realToComplex( cufftComplex *uT_I,float *poisson, int C){

  int Idx = blockIdx.x*blockDim.x + threadIdx.x;
  int Idy = blockIdx.y*blockDim.y + threadIdx.y;
  int index= Idx*C+Idy;
  if(Idx<C && Idy< C){
	  poisson_d[index]=uT_I[index].x/float(1.e+7);

  }
}





////////////////////////////////////////////////////////////////////////////////////////////////////
int main(){
	// Parametros
	L = 64.0;            // dominio de la solucion 0 <= x <= L (en longitudes de debye)
	//L=LL*LL;
	N = 10000;            // Numero de particulas
	C = 64;            // Número de celdas.
	float vb = 3.0;    // velocidad promedio de los electrones
	//double kappa = 2. * M_PI / (L);
	//float dt=0.1;    // delta tiempo (en frecuencias inversas del plasma)
	//float tmax=10000;  // cantidad de iteraciones. deben ser 100 mil segun el material
	//int skip = int (tmax / dt) / 10; //saltos del algoritmo para reportar datos
	//int itera=0;
	 float salida=0.0;
	 float dx = L / float (C);

/////////////////////////////////////////////////////////////////////////////////////////////////////
//Declaración de la posición de las particulas en x, y y velocidad en vx,vy del host y dispositivo.
	float *rx_h,*ry_h,*vx_h,*vy_h;
	float *rx_d,*ry_d, *vx_d,*vy_d;
	int *jx_d, *jy_d;
	float *yx_d;
////////////////////////////////////////////////////////////////////////////////////////////////////
	// Declaración de las variables de densidad del host y dispositivo.
	float *ne_h;
	float *ne_d;
	float *n_h;
	float *n_d;
	float *poisson_h;
	float *poisson_d;
///////////////////////////////////////////////////////////////////////////////////////////////////
	// Declaración de las variables tipo complejas con cufftComplex.
	cufftComplex *n_C, *uT_F,*uC_P,*uT_I; // n_C= Densidad Compleja. uT_F=Salida transformada rapida
									// uC_P= Salida cälculo Poisson.

////////////////////////////////////////////////////////////////////////////////////////////////////
	int size = N*sizeof(float);
	int size_ne = C*C*sizeof(float);

//////////////////////////////////////////////////////////////////////////////////////////////////////
	//reserva en memoria al host
	rx_h = (float *)malloc(size);
	ry_h = (float *)malloc(size);
	vx_h = (float *)malloc(size);
	vy_h = (float *)malloc(size);
	ne_h = (float *)malloc(size_ne);
    n_h  = (float *)malloc(size_ne);
    poisson_h  = (float *)malloc(size_ne);
//////////////////////////////////////////////////////////////////////////////////////////////////////
	//reserva de memoria del dispositivo.
	cudaMalloc((void **)&rx_d,size);
	cudaMalloc((void **)&ry_d,size);
	cudaMalloc((void **)&vx_d,size);
	cudaMalloc((void **)&vy_d,size);
	cudaMalloc((void **)&ne_d,size_ne);
	cudaMalloc((void **)&n_d,size_ne);
	cudaMalloc((void **)&jx_d,size);
	cudaMalloc((void **)&jy_d,size);
	cudaMalloc((void **)&yx_d,size);
	cudaMalloc((void **)&poisson_d,size_ne);
///////////////////////////////////////////////////////////////////////////////////////////////////////
	//reserva memoria sobre variables complejas.
	cudaMalloc((void **)&n_C,size_ne);
	cudaMalloc((void **)&uT_F,size_ne);
	cudaMalloc((void **)&uC_P,size_ne);
	cudaMalloc((void **)&uT_I,size_ne);

///////////////////////////////////////////////////////////////////////////////////////////////////////

	//valores aleatorios y tamaños de los vectores.
	curandState *devStates;
	cudaMalloc((void **) &devStates, N * sizeof(curandState));


	float blockSize = 1024;
	dim3 dimBlock (ceil(N/blockSize), 1, 1);
	dim3 dimBlock2 (ceil(C*C/blockSize), 1, 1);
	dim3 dimGrid (blockSize, 1, 1);
	int seed = time(NULL);


	distribucionParticulas<<<blockSize,dimBlock>>>(rx_d,ry_d,vx_d,vy_d,N,devStates,vb,L, seed);
	cudaDeviceSynchronize();

	inicializacionDensidad<<<blockSize,dimBlock2>>>(ne_d,C);
	cudaDeviceSynchronize();

	calculoDensidadInicializacionCeldas<<<blockSize,dimBlock>>>(rx_d,ry_d,jx_d,jy_d,yx_d,N,C,L);
	cudaDeviceSynchronize();
//
//	calculoDensidad<<<1,1>>>(ne_d,jx_d,jy_d,yx_d,C,L,N);//proceso de mejora.
//	cudaDeviceSynchronize();
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

	//funcion Calculo densidad.


	Output(ne_d,n_d,jx_d,jy_d,yx_d, C,L,N);// Calculo de la densidad y normalización de la densidad.



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//calculo de Poisson.
	realToComplex<<<blockSize,dimBlock2>>>(n_d,n_C,C);
	cufftHandle plan;
	cufftPlan2d(&plan,C,C,CUFFT_C2C);
	cufftExecC2C(plan,n_C,uT_F,CUFFT_FORWARD);
	normalizaciónSalidaUt_F<<<blockSize,dimBlock2>>>(n_C,uT_F,C);
	//calculoPoisson<<<blockSize,dimBlock2>>>(uT_F,uC_P, L, C);
	cufftExecC2C(plan,uC_P,uT_I,CUFFT_INVERSE);
	realToComplex<<<blockSize,dimBlock2>>>( uT_I,poisson_d,C);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//Asignar los datos del dispositivo al host.
	//posición en x.
	cudaMemcpy(rx_h, rx_d, size, cudaMemcpyDeviceToHost);

	// posición en y.
	cudaMemcpy(ry_h, ry_d, size, cudaMemcpyDeviceToHost);

	// velocidad en x.
	cudaMemcpy(vx_h, vx_d, size, cudaMemcpyDeviceToHost);

	//velocidad en y.
	cudaMemcpy(vy_h, vy_d, size, cudaMemcpyDeviceToHost);
	//inicializacion densidades
	cudaMemcpy(ne_h, ne_d, size_ne, cudaMemcpyDeviceToHost);
	//NormalizacionDensidades
	cudaMemcpy(n_h, n_d, size_ne, cudaMemcpyDeviceToHost);
	// Calculo de Poisson
	cudaMemcpy(poisson_h, poisson_d, size_ne, cudaMemcpyDeviceToHost);


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	//Imprimir resultados.

	ofstream init;
		init.open("distribucionInicial.txt");
		  		    for (int i = 0; i < N; i++){
		  		    	init<<rx_h[i]<<" "<<ry_h[i]<<" "<<vx_h[i]<<" "<<vy_h[i]<<endl;

		  		    }

		  		    init.close();


		init.open("salida_densidad3.txt");
					for (int i = 0; i < C*C; i++){
						init<<ne_h[i]<<" "<<endl;
						salida+=ne_h[i];
					}

					init.close();
					cout<<salida<<" "<<dx<<endl;

		init.open("densidadNormalizada.txt");
					for (int i = 0; i < C; i++){
						for (int j = 0; j < C; j++){
						init<<n_h[(i*C)+j]<<" ";
						}
					init<<endl;
					}

					init.close();
					cout<<salida<<" "<<dx<<endl;



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//Liberar memoria.
	free(rx_h);
	free(ry_h);
	free(vx_h);
	free(vy_h);
	free(ne_h);
	free(n_h);
	free(poisson_h);
	cufftDestroy(plan);
	cudaFree(rx_d);
	cudaFree(ry_d);
	cudaFree(vx_d);
	cudaFree(vy_d);
	cudaFree(ne_d);
	cudaFree(n_d);
	cudaFree(n_C);
	cudaFree(uT_F);
	cudaFree(uC_P);
	cudaFree(uT_I);
	cudaFree(poisson_d);

	return (0);

}
