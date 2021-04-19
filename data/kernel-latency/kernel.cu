#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#define USECPSEC 1000000ULL

unsigned long long dtime_usec(unsigned long long start){

  timeval tv;
  gettimeofday(&tv, 0);
  return ((tv.tv_sec*USECPSEC)+tv.tv_usec)-start;
}

__global__ void tkernel(){
}

int main(){

  tkernel<<<2000, 32>>>();
  cudaDeviceSynchronize();
  unsigned long long dt = dtime_usec(0);
  unsigned long long dt1 = dt;
  tkernel<<<2000, 32>>>();
  dt = dtime_usec(dt);
  cudaDeviceSynchronize();
  dt1 = dtime_usec(dt1);
  printf("kernel launch: %fs, kernel duration: %fs\n", dt/(float)USECPSEC, dt1/(float)USECPSEC);
}
