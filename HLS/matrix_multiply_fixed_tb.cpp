/*****************************************************************************
 *
 *     Author: Xilinx, Inc.
 *
 *     This text contains proprietary, confidential information of
 *     Xilinx, Inc. , is distributed by under license from Xilinx,
 *     Inc., and may be used, copied and/or disclosed only pursuant to
 *     the terms of a valid license agreement with Xilinx, Inc.
 *
 *     XILINX IS PROVIDING THIS DESIGN, CODE, OR INFORMATION "AS IS"
 *     AS A COURTESY TO YOU, SOLELY FOR USE IN DEVELOPING PROGRAMS AND
 *     SOLUTIONS FOR XILINX DEVICES.  BY PROVIDING THIS DESIGN, CODE,
 *     OR INFORMATION AS ONE POSSIBLE IMPLEMENTATION OF THIS FEATURE,
 *     APPLICATION OR STANDARD, XILINX IS MAKING NO REPRESENTATION
 *     THAT THIS IMPLEMENTATION IS FREE FROM ANY CLAIMS OF INFRINGEMENT,
 *     AND YOU ARE RESPONSIBLE FOR OBTAINING ANY RIGHTS YOU MAY REQUIRE
 *     FOR YOUR IMPLEMENTATION.  XILINX EXPRESSLY DISCLAIMS ANY
 *     WARRANTY WHATSOEVER WITH RESPECT TO THE ADEQUACY OF THE
 *     IMPLEMENTATION, INCLUDING BUT NOT LIMITED TO ANY WARRANTIES OR
 *     REPRESENTATIONS THAT THIS IMPLEMENTATION IS FREE FROM CLAIMS OF
 *     INFRINGEMENT, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *     FOR A PARTICULAR PURPOSE.
 *
 *     Xilinx products are not intended for use in life support appliances,
 *     devices, or systems. Use in such applications is expressly prohibited.
 *
 *     (c) Copyright 2013-2014 Xilinx Inc.
 *     All rights reserved.
 *
 *****************************************************************************/

#include "matrix_multiply_fixed.h"
#include "hls/linear_algebra/utils/x_hls_matrix_utils.h"
#include "hls/linear_algebra/utils/x_hls_matrix_tb_utils.h"
#include <iostream>
#include <fstream>
// Dummy top-level for testbench. 
// o A different synthesis top-level is selected for each solution by using set_directive_top
// o DESIGN_TOP is the function name specified as the project top (set_top) which each solution
//   points to a different implementation top-level function.
void DESIGN_TOP(MATRIX_T A [A_ROWS_F][A_COLS_F],
                MATRIX_T B [B_ROWS_F][B_COLS_F],
                      MATRIX_T C [C_ROWS_F][C_COLS_F]){
}

int main (void){

  MATRIX_T A[A_ROWS][A_COLS];
  MATRIX_T B[B_ROWS][B_COLS];
  MATRIX_T C[C_ROWS][C_COLS];
  int cholesky_sucess;

  MATRIX_T cluster_mat1[K][ANTENNA_PER_CLUSTER];
  MATRIX_T cluster_mat2[K][K];
  MATRIX_T yc[ANTENNA_PER_CLUSTER][1];
  MATRIX_T data_path1_in[K][1];
  MATRIX_T data_path2_in[K][1];
  MATRIX_T data_path1_out[K][1];
  MATRIX_T data_path2_out[K][1];
  int clust_number=0;
 
  A[0][0]=1.0;A[0][1]=0;A[0][2]=0;
  A[1][0]=2.0;A[1][1]=1.0;A[1][2]=0;
  A[2][0]=3.0;A[2][1]=4.0;A[2][2]=1.0;

  B[0][0]=2.0;
  B[1][0]=3.0;
  B[2][0]=4.0;



  std::ifstream cl_m1_r;
  std::ifstream cl_m1_i;
  std::ifstream cl_m2_r;
  std::ifstream cl_m2_i;
  std::ifstream y_r;
  std::ifstream y_i;

  cl_m1_r.open("C://workspace//PhD_work//HLS//mimo_distributed//cluster_mat1_r.txt");
  cl_m1_i.open("C://workspace//PhD_work//HLS//mimo_distributed//cluster_mat1_i.txt");
  cl_m2_r.open("C://workspace//PhD_work//HLS//mimo_distributed//cluster_mat2_r.txt");
  cl_m2_i.open("C://workspace//PhD_work//HLS//mimo_distributed//cluster_mat2_i.txt");
  y_r.open("C://workspace//PhD_work//HLS//mimo_distributed//y_r.txt");
  y_i.open("C://workspace//PhD_work//HLS//mimo_distributed//y_i.txt");

  float tempr,tempi;



  if (!cl_m1_r.is_open() || !cl_m1_i.is_open() || !cl_m2_r.is_open() || !cl_m2_r.is_open() || !y_r.is_open() || !y_i.is_open() )  // check file is open, quit if not
  {
      std::cerr << "failed to open file\n";
      std::cout << "fail to open file\n";
   }



      float tiles[3][2];

      for (int i = 0; i < K; i++) {
          for (int j = 0; j < ANTENNA_PER_CLUSTER; j++) {
        	  cl_m1_r >> tempr;
        	  cl_m1_i >> tempi;
			  cluster_mat1[i][j].real(tempr);
			  cluster_mat1[i][j].imag(tempi);

            }
          std::cout << '\n';
      }

      for (int i = 0; i < K; i++) {
               for (int j = 0; j < K; j++) {
            	   cl_m2_r >> tempr;
            	   cl_m2_i >> tempi;
            	   cluster_mat2[i][j].real(tempr);
            	   cluster_mat2[i][j].imag(tempi);
               }
               std::cout << '\n';
      }

      for (int i = 0; i < K; i++) {

    	  data_path1_in[i][0].real(i);
    	  data_path2_in[i][0].imag(i);

      }


      for (int i = 0; i < ANTENNA_PER_CLUSTER; i++) {
    	  	  	  y_r >> tempr;
    	  	  	  y_i >> tempi;
    	  	  	  yc[i][0].real(tempr);
    	  	  	  yc[i][0].imag(tempi);

                     std::cout << '\n';
       }


  for (int i=0;i<10;i++)
     sqn_ring_cluster_flatten_system_gen(cluster_mat1,cluster_mat2, yc, data_path1_in, data_path2_in, data_path1_out, data_path2_out,clust_number);




  printf("data_path1_out = \n");
  hls::print_matrix<K, 1, MATRIX_T, hls::NoTranspose>(data_path1_out, "   ");

  printf("data_path2_out = \n");
    hls::print_matrix<K, 1, MATRIX_T, hls::NoTranspose>(data_path2_out, "   ");


  printf("\nMatrix inversion status %d",cholesky_sucess);
  return(0);
}

