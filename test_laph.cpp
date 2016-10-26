/****************************************************
 * test_laph.c
 *
 * Sun May 31 17:05:41 CEST 2015
 *
 * PURPOSE:
 * TODO:
 * DONE:
 * CHANGES:
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <getopt.h>
#ifdef HAVE_MPI
#  include <mpi.h>
#endif
#ifdef HAVE_OPENMP
#include <omp.h>
#endif

#ifdef __cplusplus
extern "C"
{
#endif

#  ifdef HAVE_TMLQCD_LIBWRAPPER
#    include "tmLQCD.h"
#  endif

#ifdef __cplusplus
}
#endif

#define MAIN_PROGRAM

#include "types.h"
#include "cvc_complex.h"
#include "ilinalg.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "set_default.h"
#include "mpi_init.h"
#include "io.h"
#include "io_utils.h"
#include "propagator_io.h"
#include "gauge_io.h"
#include "read_input_parser.h"
#include "laplace_linalg.h"
#include "hyp_smear.h"
#include "laphs_io.h"
#include "laphs_utils.h"
#include "laphs.h"

using namespace cvc;


void usage(void) {
  fprintf(stdout, "usage:\n");
  exit(0);
}

int main(int argc, char **argv) {
  
  int c, mu, nu, status, sid;
  int it_src = 1;
  int is_src = 2;
  int iv_src = 3;
  int i, j, k, ncon=-1, is, idx;
  int filename_set = 0;
  int x0, x1, x2, x3, ix, iix;
  int y0, y1, y2, y3;
  int threadid, nthreads;
  int no_fields = 2;
  double dtmp[4], norm, norm2, norm3;

  double plaq=0.;
  double *gauge_field_smeared = NULL;
  int verbose = 0;
  char filename[200];
  FILE *ofs=NULL;
  double v1[6], v2[6];
  size_t items, bytes;
  complex w, w1, w2;
  double **perambulator = NULL;
  double ratime, retime;
  eigensystem_type es;
  randomvector_type rv[3];
  perambulator_type peram[3];

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?vf:")) != -1) {
    switch (c) {
    case 'v':
      verbose = 1;
      break;
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  }

  /* set the default values */
  if(filename_set==0) strcpy(filename, "cvc.input");
  if(g_cart_id==0) fprintf(stdout, "# [test_laph] Reading input from file %s\n", filename);
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [p2gg_xspace] calling tmLQCD wrapper init functions\n");

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  status = tmLQCD_invert_init(argc, argv, 1);
  if(status != 0) {
    EXIT(14);
  }
  status = tmLQCD_get_mpi_params(&g_tmLQCD_mpi);
  if(status != 0) {
    EXIT(15);
  }
  status = tmLQCD_get_lat_params(&g_tmLQCD_lat);
  if(status != 0) {
    EXIT(16);
  }
#endif

  /* initialize MPI parameters */
  mpi_init(argc, argv);

  /* initialize T etc. */
  fprintf(stdout, "# [%2d] parameters:\n"\
                  "# [%2d] T_global     = %3d\n"\
                  "# [%2d] T            = %3d\n"\
		  "# [%2d] Tstart       = %3d\n"\
                  "# [%2d] LX_global    = %3d\n"\
                  "# [%2d] LX           = %3d\n"\
		  "# [%2d] LXstart      = %3d\n"\
                  "# [%2d] LY_global    = %3d\n"\
                  "# [%2d] LY           = %3d\n"\
		  "# [%2d] LYstart      = %3d\n",\
		  g_cart_id, g_cart_id, T_global, g_cart_id, T, g_cart_id, Tstart,
		             g_cart_id, LX_global, g_cart_id, LX, g_cart_id, LXstart,
		             g_cart_id, LY_global, g_cart_id, LY, g_cart_id, LYstart);

  if(init_geometry() != 0) {
    fprintf(stderr, "[test_laph] ERROR from init_geometry\n");
    exit(101);
  }

  geometry();


  /* read the gauge field */
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
  if(g_cart_id==0) fprintf(stdout, "# [test_laph] reading gauge field from file %s\n", filename);

  if(strcmp(gaugefilename_prefix,"identity")==0) {
    status = unit_gauge_field(g_gauge_field, VOLUME);
  } else {
    // status = read_nersc_gauge_field_3x3(g_gauge_field, filename, &plaq);
    // status = read_ildg_nersc_gauge_field(g_gauge_field, filename);
    status = read_lime_gauge_field_doubleprec(filename);
    // status = read_nersc_gauge_field(g_gauge_field, filename, &plaq);
  }
  if(status != 0) {
    fprintf(stderr, "[test_laph] Error, could not read gauge field\n");
    exit(11);
  }
  // measure the plaquette
  if(g_cart_id==0) fprintf(stdout, "# [test_laph] read plaquette value 1st field: %25.16e\n", plaq);
  plaquette(&plaq);
  if(g_cart_id==0) fprintf(stdout, "# [test_laph] measured plaquette value 1st field: %25.16e\n", plaq);

#if 0
  /* smear the gauge field */
  status = hyp_smear_3d (g_gauge_field, N_hyp, alpha_hyp, 0, 0);
  if(status != 0) {
    fprintf(stderr, "[test_laph] Error from hyp_smear_3d, status was %d\n", status);
    EXIT(7);
  }

  plaquette(&plaq);
  if(g_cart_id==0) fprintf(stdout, "# [test_laph] measured plaquette value ofter hyp smearing = %25.16e\n", plaq);

  sprintf(filename, "%s_hyp.%.4d", gaugefilename_prefix, Nconf);
  fprintf(stdout, "# [test_laph] writing hyp-smeared gauge field to file %s\n", filename);

  status = write_lime_gauge_field(filename, plaq, Nconf, 64);
  if(status != 0) {
    fprintf(stderr, "[apply_lapace] Error friom write_lime_gauge_field, status was %d\n", status);
    EXIT(7);
  }
#endif

  /* init and allocate spinor fields */
  no_fields = 2;
  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields; i++) alloc_spinor_field(&g_spinor_field[i], VOLUME+RAND);



  init_eigensystem(&es);
  init_perambulator(&peram);
  init_randomvector(&rv);

  status = alloc_eigensystem (&es, T, laphs_eigenvector_number);
  if(status != 0) {
    fprintf(stderr, "[test_laph] Error from alloc_eigensystem, status was %d\n", status);
    EXIT(7);
  }

  ratime = _GET_TIME;
  status = read_eigensystem(&es);
  if (status != 0) {
    fprintf(stderr, "# [test_laph] Error from read_eigensystem, status was %d\n", status);
  }
  retime = _GET_TIME;
  fprintf(stdout, "# [] time to read eigensystem %e\n", retime-ratime);

/*
  ratime = _GET_TIME;
  status = test_eigensystem(&es, g_gauge_field);
  retime = _GET_TIME;
  fprintf(stdout, "# [] time to test eigensystem %e\n", retime-ratime);
*/

  status = alloc_randomvector(&rv, T, 4, laphs_eigenvector_number);
  status = alloc_randomvector(&prv, T, 4, laphs_eigenvector_number);
  status = read_randomvector(&rv, "u", 0);

  /*****************************************************************
   * project the random vector
   *****************************************************************/
  ratime = _GET_TIME;
  status = project_randomvector (&prv, &rv, it_src, is_src, iv_src);
  if(status != 0) {
    fprintf(stderr, "[test_laph] Error from project_randomvector, status was %d\n", status);
    EXIT(8);
  }
  retime = _GET_TIME;
  fprintf(stdout, "# [] time to project randomvector %e\n", retime-ratime);

  /* TEST */
  sprintf(filename, "projected_randomvector.t%d_s%d_v%d.ascii", it_src, is_src, iv_src);
  ofs = fopen(filename, "w");
  ix=0; 
  for(x0=0; x0<prv.nt; x0++) {
    for(i=0; i<prv.ns; i++) {
      for(k=0; k<prv.nv; k++) {
        fprintf(ofs, "%3d%3d%5d%25.16e%25.16e\n", x0, i, k, prv.rvec[2*ix], prv.rvec[2*ix+1]);
        ix++;
      }
    }
  }
  fclose(ofs);
#if 0
  /* print_randomvector(&rv, stdout); */
  sprintf(filename, "projected_randomvector.t%d_s%d_v%d", it_src, is_src, iv_src);
  ofs = fopen(filename, "w");
  print_randomvector(&prv, ofs);
  fclose(ofs);

  /*****************************************************************
   * prepare the source for the inversion
   *****************************************************************/
  ratime = _GET_TIME;
  status = fv_eq_eigensystem_ti_randomvector (g_spinor_field[0], &es, &prv);
  if(status != 0) {
    fprintf(stderr, "[test_laph] Error from , status was %d\n", status);
    EXIT(9);
  }
  retime = _GET_TIME;
  fprintf(stdout, "# [] time for fv = V x r %e\n", retime-ratime);

  sprintf(filename, "v_projected_randomvector.t%d_s%d_v%d", it_src, is_src, iv_src);
  status = write_propagator(g_spinor_field[0], filename, 0, 64);


  /* TEST */
  /* write ascii file */
  sprintf(filename, "v_projected_randomvector.t%d_s%d_v%d.ascii", it_src, is_src, iv_src);
  ofs = fopen(filename, "w");
  for(x0=0; x0<T; x0++) {
    for(x1=0; x1<LX; x1++) {
    for(x2=0; x2<LX; x2++) {
    for(x3=0; x3<LX; x3++) {
      ix = g_ipt[x0][x1][x2][x3];
      iix = _GSI(ix);
      fprintf(stdout, "")
      for(i=0; i<12; i++) {
        fprintf(ofs, "%3d%3d%26.16e%25.16e\n", i/3, i%3, )
      }


    }}}
  }
  fclose(ofs);
#endif

/*
  sprintf(filename, "v_projected_randomvector.t%d_s%d_v%d.ascii", it_src, is_src, iv_src);
  ofs = fopen(filename, "w");
  status = printf_spinor_field(g_spinor_field[0], 1, ofs);
  fclose(ofs);
*/
  /* invert with V P rv as the source */


  /* contract with eigensystem V^+ S V P rv */
    

#if 0
  status = alloc_perambulator(&peram, laphs_time_src_number, laphs_spin_src_number, laphs_evec_src_number, T, 4, laphs_eigenvector_number, 3, "u", "smeared1", 0 );

  print_perambulator_info (&peram);

  status = read_perambulator(&peram);
  if(status != 0) {
    fprintf(stderr, "[test_laph] Error from read_perambulator, status was %d\n", status);
    exit(12);
  }

#endif


  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/

  fini_eigensystem (&es);
  fini_randomvector(&rv);
  fini_perambulator(&peram);


  free(g_gauge_field);
  for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
  free(g_spinor_field);

  free_geometry();

  g_the_time = time(NULL);
  fprintf(stdout, "# [test_laph] %s# [test_laph] end fo run\n", ctime(&g_the_time));
  fflush(stdout);
  fprintf(stderr, "# [test_laph] %s# [test_laph] end fo run\n", ctime(&g_the_time));
  fflush(stderr);


#ifdef HAVE_MPI
  MPI_Finalize();
#endif
  return(0);
}

