//Author: Enliang Li
//mian translation for the forward algorithm (gpu-version)
//Latest Version: 1.2 on Apr.15th 2019

#include "forward.cuh"

//Define any useful program-wide constants here
#define x_dim 11
#define y_dim 40
#define batch 4
#define states 3


__global__ void pair_HMM_forward(
  const int cur_i,
  const int cur_j,
  double forward_matrix_in[x_dim+1][y_dim+1][batch][2],
  double transitions[x_dim+1][batch][2][3],
  double emissions[x_dim+1][y_dim+1][batch][2],
  double m[2][2][batch][2],
  double start_transitions[batch][2],
  double forward_matrix_out[x_dim+1][y_dim+1][batch][2]
) {

  int batch_id = blockIdx.x;
  int states_id = threadIdx.x;

  __shared__ double e[batch][2];
  e[batch_id][states_id] = emissions[cur_i][cur_j][batch_id][states_id];

  __syncthreads();

  double t[2][2][batch][2][2];
  for (int k = 0; k < 2; k++) {
    for (int l = 0; l < 2; l++) {
      t[0][0][batch_id][k][l] = transitions[cur_i - 1][batch_id][k][l];
      t[0][1][batch_id][k][l] = transitions[cur_i - 1][batch_id][k][l];
      t[1][0][batch_id][k][l] = transitions[cur_i][batch_id][k][l];
      t[1][1][batch_id][k][l] = transitions[cur_i][batch_id][k][l];
    }
  }
  __syncthreads();

  if (cur_i > 0 && cur_j == 0) {
    if (cur_i == 1) {

      forward_matrix_out[1][0][batch_id][states_id] = start_transitions[batch][states_id] * e[0][states_id];

    } else {

      double t01[batch][2][2];
      for (int j = 0; j < 2; j++) {
        for (int k = 0; k < 2; k++) {
          t01[batch_id][j][k] = t[0][1][batch_id][j][k];
        }
      }

      __shared__ double f01[1][batch][2];
      f01[0][batch_id][states_id] = forward_matrix_in[cur_i - 1][cur_j][batch_id][states_id];

      __shared__ double multiplication_temp[1][batch][2];
      double new_val = 0.0;
      new_val = 0.0;
      for (int k = 0; k < 2; k++) {
        new_val += f01[0][batch_id][k] * t01[batch_id][k][states_id];
      }
      new_val *= (e[batch_id][states_id] * m[0][1][batch_id][states_id]);
      multiplication_temp[0][batch_id][states_id] = new_val;


      forward_matrix_out[cur_i][0][batch_id][states_id] = multiplication_temp[0][batch_id][states_id];

    }
  }
  else if (cur_i > 0 and cur_j > 0) {

    double f[2][2][batch][1][2];
    for (int i = 0; i < 2; i++) {
      f[0][0][batch_id][0][i] = forward_matrix_in[cur_i-1][cur_j-1][batch_id][i];
      f[0][1][batch_id][0][i] = forward_matrix_in[cur_i-1][cur_j][batch_id][i];
      f[1][0][batch_id][0][i] = forward_matrix_in[cur_i][cur_j-1][batch_id][i];
      f[1][1][batch_id][0][i] = forward_matrix_in[cur_i][cur_j][batch_id][i];
    }

    __shared__ double multiplication_temp[4][batch][1][2];
    double new_val0 = 0.0;
    double new_val1 = 0.0;
    double new_val2 = 0.0;
    double new_val3 = 0.0;

    for (int k = 0; k < 2; k++) {
      new_val0 += f[0][0][batch_id][0][k] * t[0][0][batch_id][k][states_id];
      new_val1 += f[0][1][batch_id][0][k] * t[0][1][batch_id][k][states_id];
      new_val2 += f[1][0][batch_id][0][k] * t[1][0][batch_id][k][states_id];
      new_val3 += f[1][1][batch_id][0][k] * t[1][1][batch_id][k][states_id];
    }
    new_val0 *= m[0][0][batch_id][states_id];
    new_val1 *= m[0][1][batch_id][states_id];
    new_val2 *= m[1][0][batch_id][states_id];
    new_val3 *= m[1][1][batch_id][states_id];
    multiplication_temp[0][batch_id][0][states_id] = new_val0;
    multiplication_temp[1][batch_id][0][states_id] = new_val1;
    multiplication_temp[2][batch_id][0][states_id] = new_val2;
    multiplication_temp[3][batch_id][0][states_id] = new_val3;

    //double forward_each_update[1][2];
    for (int j = 0; j < 2; j++) {
      double summation = 0.0;

      summation = multiplication_temp[0][batch_id][0][j] + multiplication_temp[1][batch_id][0][j] +
      multiplication_temp[2][batch_id][0][j] +
      multiplication_temp[3][batch_id][0][j];

      summation *= e[batch_id][j];

      forward_matrix_out[cur_i][cur_j][batch_id][j] = summation;
    }

  }

  return;
}




int main() {

  double dev_cur_forward[x_dim+1][y_dim+1][batch][states-1];
  double dev_trans[x_dim+1][batch][states-1][states];
  double dev_emis[x_dim+1][y_dim+1][batch][states-1];
  double dev_like[2][2][batch][states-1];
  double dev_start[batch][states-1];
  double dev_next_forward[x_dim+1][y_dim+1][batch][states-1];

  dim3 dimBlock((states-1), 1, 1);
  dim3 dimGrid(batch, 1, 1);

  size_t forward_matrix_size = (x_dim+1)*(y_dim+1)*batch*(states-1)*sizeof(double);
  size_t emissions_size = (x_dim+1)*(y_dim+1)*batch*(states-1)*sizeof(double);
  size_t transitions_size = (x_dim+1)*(states-1)*states*batch*sizeof(double);
  size_t start_transitions_size = batch*(states-1)*sizeof(double);
  size_t likelihood_size = 2*2*(states-1)*batch*sizeof(double);

  cudaMalloc((void**)&dev_cur_forward, forward_matrix_size);
  cudaMalloc((void**)&dev_next_forward, forward_matrix_size);
  cudaMalloc((void**)&dev_emis, emissions_size);
  cudaMalloc((void**)&dev_trans, transitions_size);
  cudaMalloc((void**)&dev_like, likelihood_size);
  cudaMalloc((void**)&dev_start, start_transitions_size);

  auto t1 = std::chrono::high_resolution_clock::now();

  for(int count = 0; count < 10 ;count++) {
    for (int i = 0; i < x_dim + 1; i++) {
      for (int j = 0; j < y_dim + 1; j++) {
        pair_HMM_forward<<<dimGrid, dimBlock>>>(i, j, dev_cur_forward, dev_trans, dev_emis, dev_like, dev_start, dev_next_forward);
        cudaDeviceSynchronize();
        cudaMemcpy(dev_cur_forward, dev_next_forward, forward_matrix_size, cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
      }
    }

  }

  auto t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> milli = (t2 - t1);
  std::cout << "pair HMM took " <<  milli.count() << " milliseconds\n" ;

  cudaFree(dev_cur_forward);
  cudaFree(dev_next_forward);
  cudaFree(dev_emis);
  cudaFree(dev_trans);
  cudaFree(dev_like);
  cudaFree(dev_start);

  return 1;

}
