/***************************************************************************
 *
 * njjn_w_pc_charged_gf_invert_contract_smear_hb
 *
 ***************************************************************************/

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <complex.h>
#ifdef HAVE_MPI
#  include <mpi.h>
#endif
#ifdef HAVE_OPENMP
#  include <omp.h>
#endif
#include <getopt.h>

#ifdef HAVE_LHPC_AFF
#include "lhpc-aff.h"
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

#include "cvc_complex.h"
#include "cvc_linalg.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "cvc_timer.h"
#include "mpi_init.h"
#include "set_default.h"
#include "io.h"
#include "propagator_io.h"
#include "read_input_parser.h"
#include "smearing_techniques.h"
#include "contractions_io.h"
#include "Q_clover_phi.h"
#include "prepare_source.h"
#include "prepare_propagator.h"
#include "project.h"
#include "table_init_z.h"
#include "table_init_d.h"
#include "dummy_solver.h"
#include "contractions_io.h"
#include "contract_factorized.h"
#include "contract_diagrams.h"
#include "gamma.h"
#include "clover.h"
#include "fermion_quda.h"
#include "gauge_quda.h"

#define MAX_NUM_GF_NSTEP 100

#define _OP_ID_UP 0
#define _OP_ID_DN 1

#define _PART_PROP    1
#define _PART_TWOP    1  /* N1, N2 */
#define _PART_THREEP  1  /* W type sequential diagrams */

#ifndef _TEST_TIMER
#  define _TEST_TIMER
#endif


using namespace cvc;

/* typedef int ( * reduction_operation ) (double**, double*, fermion_propagator_type*, unsigned int ); */

typedef int ( * reduction_operation ) (double**, fermion_propagator_type*, fermion_propagator_type*, fermion_propagator_type*, unsigned int);


/***************************************************************************
 * 
 ***************************************************************************/
static inline int reduce_project_write ( double ** vx, double *** vp, fermion_propagator_type * fa, fermion_propagator_type * fb, fermion_propagator_type * fc, reduction_operation reduce,
    void * writer, char * tag, int (*momentum_list)[3], int momentum_number, int const nd, unsigned int const N, int const io_proc ) {

  int exitstatus;

  /* contraction */
  exitstatus = reduce ( vx, fa, fb, fc, N );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[reduce_project_write] Error from reduce, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    return( 1 );
  }

  /* (partial) Fourier transform, projection from position space to a (small) subset of momentum space */
  exitstatus = contract_vn_momentum_projection ( vp, vx, nd, momentum_list, momentum_number);
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[reduce_project_write] Error from contract_vn_momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    return( 2 );
  }

#if defined HAVE_HDF5
  exitstatus = contract_vn_write_h5 ( vp, nd, (char *)writer, tag, momentum_list, momentum_number, io_proc );
#elif defined HAVE_LHPC_AFF
  /* write to AFF file */
  exitstatus = contract_vn_write_aff ( vp, nd, (struct AffWriter_s*)writer, tag, momentum_list, momentum_number, io_proc );
#endif
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[reduce_project_write] Error from contract_vn_write for tag %s, status was %d %s %d\n", tag, exitstatus, __FILE__, __LINE__ );
    return( 3 );
  }

  return ( 0 );

}  /* end of reduce_project_write */


/***************************************************************************
 * 
 ***************************************************************************/
static inline int project_write ( double ** vx, double *** vp, void * writer, char * tag, int (*momentum_list)[3], int momentum_number, int const nd, int const io_proc ) {

  int exitstatus;

  /* (partial) Fourier transform, projection from position space to a (small) subset of momentum space */
  exitstatus = contract_vn_momentum_projection ( vp, vx, nd, momentum_list, momentum_number);
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[reduce_project_write] Error from contract_vn_momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    return( 2 );
  }

#if defined HAVE_HDF5
  exitstatus = contract_vn_write_h5 ( vp, nd, (char *)writer, tag, momentum_list, momentum_number, io_proc );
#elif defined HAVE_LHPC_AFF
  /* write to AFF file */
  exitstatus = contract_vn_write_aff ( vp, nd, (struct AffWriter_s *)writer, tag, momentum_list, momentum_number, io_proc );
#endif
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[reduce_project_write] Error from contract_vn_write for tag %s, status was %d %s %d\n", tag, exitstatus, __FILE__, __LINE__ );
    return( 3 );
  }

  return ( 0 );

}  /* end of reduce_project_write */

/***************************************************************************
 * helper message
 ***************************************************************************/
void usage() {
  fprintf(stdout, "Code for FHT-type nucleon-nucleon 2-pt function inversion and contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f input <filename> : input filename for [default cpff.input]\n");
  fprintf(stdout, "          -c                  : check propagator residual [default false]\n");
  fprintf(stdout, "          -h                  : this message\n");
  EXIT(0);
}

/***************************************************************************
 *
 * MAIN PROGRAM
 *
 ***************************************************************************/
int main(int argc, char **argv) {
  
  const char outfile_prefix[] = "njjn_w_pc_charged_gf";

  const char flavor_tag[4] = { 'u', 'd', 's', 'c' };

  const int sequential_gamma_sets = 2;
  int const sequential_gamma_num[2] = {4, 4};
  int const sequential_gamma_id[2][4] = {
    { 0,  1,  2,  3 },
    { 6,  7,  8,  9 } };

  char const gamma_id_to_Cg_ascii[16][10] = {
    "Cgy",
    "Cgzg5",
    "Cgt",
    "Cgxg5",
    "Cgygt",
    "Cgyg5gt",
    "Cgyg5",
    "Cgz",
    "Cg5gt",
    "Cgx",
    "Cgzg5gt",
    "C",
    "Cgxg5gt",
    "Cgxgt",
    "Cg5",
    "Cgzgt"
  };


  char const gamma_id_to_ascii[16][10] = {
    "gt",
    "gx",
    "gy",
    "gz",
    "id",
    "g5",
    "gtg5",
    "gxg5",
    "gyg5",
    "gzg5",
    "gtgx",
    "gtgy",
    "gtgz",
    "gxgy",
    "gxgz",
    "gygz" 
  };

  int gf_nstep = 0;
  int gf_niter_list[MAX_NUM_GF_NSTEP];
  double gf_dt_list[MAX_NUM_GF_NSTEP];

  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  int check_propagator_residual = 0;
  char filename[400];
  double **lmzz[2] = { NULL, NULL }, **lmzzinv[2] = { NULL, NULL };
  double *gauge_field_with_phase = NULL;
  double *gauge_field_smeared = NULL;
  struct timeval ta, tb, start_time, end_time;

  /*
  int const    gamma_f1_number                           = 4;
  int const    gamma_f1_list[gamma_f1_number]            = { 14 , 11,  8,  2 };
  double const gamma_f1_sign[gamma_f1_number]            = { +1 , +1, -1, -1 };
  */

  int const    gamma_f1_number                           = 1;
  int const    gamma_f1_list[gamma_f1_number]            = { 14 };
  double const gamma_f1_sign[gamma_f1_number]            = { +1 };

  int read_scalar_field  = 0;
  int write_scalar_field = 0;

#if ( defined HAVE_LHPC_AFF ) &&  ! (defined HAVE_HDF5 )
  struct AffWriter_s *affw = NULL;
  char aff_tag[400];
#endif

  int first_solve_dummy = 1;

  /* for gradient flow */
  int gf_niter = 1;
  int gf_ns = 1;
  double gf_dt = 0.01;
  double gf_tau = 0.;
  int gf_nb;


#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "sSch?f:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'c':
      check_propagator_residual = 1;
      break;
    case 's':
      read_scalar_field = 1;
      break;
    case 'S':
      write_scalar_field = 1;
      break;
    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  }

  gettimeofday ( &start_time, (struct timezone *)NULL );


  /***************************************************************************
   * read input and set the default values
   ***************************************************************************/
  if(filename_set==0) strcpy(filename, "twopt.input");
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [njjn_w_pc_charged_gf_invert_contract] calling tmLQCD wrapper init functions\n");

  /***************************************************************************
   * initialize tmLQCD solvers
   ***************************************************************************/
  exitstatus = tmLQCD_invert_init(argc, argv, 1, 0);
  if(exitstatus != 0) {
    EXIT(1);
  }
  exitstatus = tmLQCD_get_mpi_params(&g_tmLQCD_mpi);
  if(exitstatus != 0) {
    EXIT(2);
  }
  exitstatus = tmLQCD_get_lat_params(&g_tmLQCD_lat);
  if(exitstatus != 0) {
    EXIT(3);
  }
#endif

  /***************************************************************************
   * initialize MPI parameters for cvc
   ***************************************************************************/
  mpi_init(argc, argv);
  mpi_init_xchange_contraction(2);

  /***************************************************************************
   * report git version
   * make sure the version running here has been commited before program call
   ***************************************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [njjn_w_pc_charged_gf_invert_contract] git version = %s\n", g_gitversion);
  }

  /***************************************************************************
   * set number of openmp threads
   *
   *   each process and thread reports
   ***************************************************************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [njjn_w_pc_charged_gf_invert_contract] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [njjn_w_pc_charged_gf_invert_contract] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[njjn_w_pc_charged_gf_invert_contract] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[njjn_w_pc_charged_gf_invert_contract] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  
  /***************************************************************************
   * initialize lattice geometry
   *
   * allocate and fill geometry arrays
   ***************************************************************************/
  geometry();


  /***************************************************************************
   * set up some mpi exchangers for
   * (1) even-odd decomposed spinor field
   * (2) even-odd decomposed propagator field
   ***************************************************************************/
  mpi_init_xchange_eo_spinor();
  mpi_init_xchange_eo_propagator();

  /***************************************************************************
   * set up the gauge field
   *
   *   either read it from file or get it from tmLQCD interface
   *
   *   lime format is used
   ***************************************************************************/
#ifndef HAVE_TMLQCD_LIBWRAPPER
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  if(!(strcmp(gaugefilename_prefix,"identity")==0)) {
    /* read the gauge field */
    sprintf ( filename, "%s.%.4d", gaugefilename_prefix, Nconf );
    if(g_cart_id==0) fprintf(stdout, "# [njjn_w_pc_charged_gf_invert_contract] reading gauge field from file %s\n", filename);
    exitstatus = read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [njjn_w_pc_charged_gf_invert_contract] initializing unit matrices\n");
    exitstatus = unit_gauge_field ( g_gauge_field, VOLUME );
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[njjn_w_pc_charged_gf_invert_contract] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(5);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer( &g_gauge_field );
  if(exitstatus != 0) {
    EXIT(6);
  }
  // exitstatus = my_gauge_field ( g_gauge_field, VOLUME );
  if( g_gauge_field == NULL) {
    fprintf(stderr, "[njjn_w_pc_charged_gf_invert_contract] Error, g_gauge_field is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(7);
  }
#endif

  /***********************************************************
   * multiply the phase to the gauge field
   ***********************************************************/
  exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_phase_up );
  if(exitstatus != 0) {
    fprintf(stderr, "[njjn_w_pc_charged_gf_invert_contract] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***************************************************************************
   * initialize the clover term, 
   * lmzz and lmzzinv
   *
   *   mzz = space-time diagonal part of the Dirac matrix
   *   l   = light quark mass
   ***************************************************************************/
  exitstatus = init_clover ( &g_clover, &lmzz, &lmzzinv, gauge_field_with_phase, g_mu, g_csw );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[njjn_w_pc_charged_gf_invert_contract] Error from init_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }

  /***************************************************************************
   * set io process
   ***************************************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[njjn_w_pc_charged_gf_invert_contract] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [njjn_w_pc_charged_gf_invert_contract] proc%.4d has io proc id %d\n", g_cart_id, io_proc );

  /***************************************************************************
   * set the gamma matrices
   ***************************************************************************/
  gamma_matrix_type sequential_gamma_list[4][4];
  /* vector */
  gamma_matrix_set( &( sequential_gamma_list[0][0] ), 0, 1. );  /*  gamma_0 = gamma_t */
  gamma_matrix_set( &( sequential_gamma_list[0][1] ), 1, 1. );  /*  gamma_1 = gamma_x */
  gamma_matrix_set( &( sequential_gamma_list[0][2] ), 2, 1. );  /*  gamma_2 = gamma_y */
  gamma_matrix_set( &( sequential_gamma_list[0][3] ), 3, 1. );  /*  gamma_3 = gamma_z */
  /* pseudovector */
  gamma_matrix_set( &( sequential_gamma_list[1][0] ), 6, 1. );  /*  gamma_6 = gamma_5 gamma_t */
  gamma_matrix_set( &( sequential_gamma_list[1][1] ), 7, 1. );  /*  gamma_7 = gamma_5 gamma_x */
  gamma_matrix_set( &( sequential_gamma_list[1][2] ), 8, 1. );  /*  gamma_8 = gamma_5 gamma_y */
  gamma_matrix_set( &( sequential_gamma_list[1][3] ), 9, 1. );  /*  gamma_9 = gamma_5 gamma_z */
  /* scalar */
  gamma_matrix_set( &( sequential_gamma_list[2][0] ), 4, 1. );  /*  gamma_4 = id */
  /* pseudoscalar */
  gamma_matrix_set( &( sequential_gamma_list[3][0] ), 5, 1. );  /*  gamma_5 */

  gamma_matrix_type gammafive;
  gamma_matrix_set( &gammafive, 5, 1. );  /*  gamma_5 */


  /***************************************************************************
   * prepare the Fourier phase field
   ***************************************************************************/
  size_t const sizeof_spinor_field = _GSI( VOLUME ) * sizeof( double );

  /***************************************************************************
   * init rng state
   ***************************************************************************/
  exitstatus = init_rng_stat_file ( g_seed, NULL );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[njjn_w_pc_charged_gf_invert_contract] Error from init_rng_stat_file %s %d\n", __FILE__, __LINE__ );;
    EXIT( 50 );
  }

  /***************************************************************************
   ***************************************************************************
   **
   ** Part I
   **
   ** prepare stochastic sources for W-type sequential sources and propagators
   **
   ***************************************************************************
   ***************************************************************************/


#ifdef _GFLOW_QUDA

  /***************************************************************************
   * gauge_param initialization
   ***************************************************************************/
  double ** h_gauge = init_2level_dtable ( 4, 18*VOLUME );
  if ( h_gauge == NULL )
  {
    fprintf(stderr, "[test_adjoint_flow] Error from init_2level_dtable   %s %d\n", __FILE__, __LINE__);
    EXIT(12);
  }

  QudaGaugeParam gauge_param;
  init_gauge_param ( &gauge_param );

  /***************************************************************************
   * prepare upload of gauge field for gradient flow
   ***************************************************************************/
  /* reshape gauge field */
  gauge_field_cvc_to_qdp ( h_gauge, gauge_field_with_phase );

  gauge_param.location = QUDA_CPU_FIELD_LOCATION;

  /***************************************************************************
   * either we upload QUDA_WILSON_LINKS explicitly here, or we make
   * a dummy solve to have wrapper upload the original gauge field
   * and have quda instantiate gaugePrecise, which we need in
   * adjoint gradient flow ahead of inversion
   *
   * here we make a dummy solve
   ***************************************************************************/
  if ( first_solve_dummy == 0 )
  {
    // set to Wilson links for this one upload
    gauge_param.type = QUDA_WILSON_LINKS;
    loadGaugeQuda ( (void *)h_gauge, &gauge_param );

  } else if ( first_solve_dummy == 1 )
  {
    double ** spinor_work = init_2level_dtable ( 2, _GSI( VOLUME+RAND ) );
    if ( spinor_work == NULL ) 
    {
      fprintf ( stderr, "[njjn_bd_charged_gf_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT(12);
    }
    memset(spinor_work[1], 0, sizeof_spinor_field);
    memset(spinor_work[0], 0, sizeof_spinor_field);
    if ( g_cart_id == 0 ) spinor_work[0][0] = 1.;
    exitstatus = _TMLQCD_INVERT(spinor_work[1], spinor_work[0], _OP_ID_UP);
#  if ( defined GPU_DIRECT_SOLVER )
    if(exitstatus < 0)
#  else
    if(exitstatus != 0)
#  endif
    {
      fprintf(stderr, "[njjn_bd_charged_invert_contract] Error from invert, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(12);
    }
    fini_2level_dtable ( &spinor_work );

  }  // end of if first_solve_dummy


  // Haobo: copied from njjn_bd_charged_invert_contract.cpp
  /***********************************************
   * if we want to use Jacobi smearing, we need 
   * smeared gauge field
   ***********************************************/
  if( N_Jacobi > 0 ) {

#ifndef _SMEAR_QUDA 

    alloc_gauge_field ( &gauge_field_smeared, VOLUMEPLUSRAND);

    memcpy ( gauge_field_smeared, g_gauge_field, 72*VOLUME*sizeof(double));

    if ( N_ape > 0 ) {
#endif
      exitstatus = APE_Smearing(gauge_field_smeared, alpha_ape, N_ape);
      if(exitstatus != 0) {
        fprintf(stderr, "[njjn_bd_charged_invert_contract] Error from APE_Smearing, status was %d\n", exitstatus);
        EXIT(47);
      }
#ifndef _SMEAR_QUDA 
    }  /* end of if N_ape > 0 */

    /***********************************************
     * check plaquette value after APE smearing
     *
     * ONLY IF NOT SMEARING ON DEVICE
     * in case of smearing on device, there is
     * not any non-NULL smeared gauge field 
     * pointer on host
     ***********************************************/
    exitstatus = plaquetteria( gauge_field_smeared );
    if( exitstatus != 0 ) {
      fprintf(stderr, "[njjn_bd_charged_invert_contract] Error from plaquetteria, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(2);
    }
#endif
  }  /* end of if N_Jacobi > 0 */


  gf_nstep = 3;
  double gf_dt_fixed = 0.01;
  gf_niter_list[0] = 0;
  for( int i = 1; i < gf_nstep; i++ )
  {
    gf_niter_list[i] = 1;
  }


  /***************************************************************************
   * set to flowed links from now on, upload gauge field
   ***************************************************************************/
  gauge_param.type = QUDA_FLOWED_LINKS;
#ifdef _TEST_TIMER
  gettimeofday ( &ta, (struct timezone *)NULL );
#endif
  loadGaugeQuda ( (void *)h_gauge, &gauge_param );
#ifdef _TEST_TIMER
  gettimeofday ( &tb, (struct timezone *)NULL );
  show_time ( &ta, &tb, "njjn_bd_charged_gf_invert_contract", "loadGaugeQuda", g_cart_id == 0 );
#endif

  if ( io_proc == 2 && g_verbose > 1 )
  {
    fprintf (stdout, "# [njjn_bd_charged_gf_invert_contract] gf_nb = %d   %s %d\n", gf_nb, __FILE__, __LINE__ );
  }

#endif  // of if _GFLOW_QUDA

  /***************************************************************************
   * loop on source locations
   *
   *   each source location is given by 4-coordinates in
   *   global variable
   *   g_source_coords_list[count][0-3] for t,x,y,z
   ***************************************************************************/
  for( int isource_location = 0; isource_location < g_source_location_number; isource_location++ )
  {
    /***************************************************************************
     * allocate point-to-all propagators,
     * spin-color dilution (i.e. 12 fields per flavor of size 24xVOLUME real )
     ***************************************************************************/

    /***************************************************************************
     * determine source coordinates,
     * find out, if source_location is in this process
     ***************************************************************************/

    int const gsx[4] = {
        ( g_source_coords_list[isource_location][0] +  T_global ) %  T_global,
        ( g_source_coords_list[isource_location][1] + LX_global ) % LX_global,
        ( g_source_coords_list[isource_location][2] + LY_global ) % LY_global,
        ( g_source_coords_list[isource_location][3] + LZ_global ) % LZ_global };

    int sx[4], source_proc_id = -1;
    exitstatus = get_point_source_info (gsx, sx, &source_proc_id);
    if( exitstatus != 0 ) {
      fprintf(stderr, "[njjn_w_pc_charged_gf_invert_contract] Error from get_point_source_info status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(123);
    }

    double * point_source_flowed = init_1level_dtable ( _GSI(VOLUME) );
    if ( point_source_flowed == NULL )
    {
      fprintf ( stderr, "[njjn_fht_gf_invert_contract] Error from init_level_table    %s %d\n", __FILE__, __LINE__ );
      EXIT(2);
    }





      point_source_flowed[_GSI(g_ipt[sx[0]][sx[1]][sx[2]][sx[3]])] = 1.;
      QudaGaugeSmearParam smear_param;
    gf_niter = 20;
    gf_ns = 1;
    gf_dt = 0.01;
    gf_tau = 0.;
    gf_nb = (int) ceil ( pow ( (double)gf_niter , 1./ ( (double)gf_ns + 1. ) ) );
      smear_param.n_steps       = gf_niter;
      smear_param.epsilon       = gf_dt;
      smear_param.meas_interval = 1;
      smear_param.smear_type    = QUDA_GAUGE_SMEAR_WILSON_FLOW;
      smear_param.restart       = QUDA_BOOLEAN_TRUE;
      gettimeofday ( &ta, (struct timezone *)NULL );
      _performGFlowAdjoint ( point_source_flowed, point_source_flowed, &smear_param, gf_niter, gf_nb, gf_ns );
      gettimeofday ( &tb, (struct timezone *)NULL );
      show_time ( &ta, &tb, "njjn_w_pc_charged_gf_invert_contract", "_performGFlowAdjoint-000", g_cart_id == 0 );


    for ( int isc = 0; isc < 12; isc++ )
    {
      memset ( point_source_flowed , 0, sizeof_spinor_field );
      if ( g_cart_id == source_proc_id )
      {
        if ( g_verbose > 2 )
        {
          fprintf (stdout, "# [njjn_w_pc_charged_gf_invert_contract] proc%.4d sets the source    %s %d\n", g_cart_id, __FILE__, __LINE__ );
        }
        // have process for source position, fill in the source vector
        point_source_flowed[_GSI(g_ipt[sx[0]][sx[1]][sx[2]][sx[3]])+2*isc] = 1.;
      }

#ifdef _GFLOW_QUDA
      /***********************************************************
       * apply adjoint flow to the point source
       ***********************************************************/
    gf_niter = 20;
    gf_ns = 1;
    gf_dt = 0.01;
    gf_tau = 0.;
    gf_nb = (int) ceil ( pow ( (double)gf_niter , 1./ ( (double)gf_ns + 1. ) ) );
      smear_param.n_steps       = gf_niter;
      smear_param.epsilon       = gf_dt;
      smear_param.meas_interval = 1;
      smear_param.smear_type    = QUDA_GAUGE_SMEAR_WILSON_FLOW;
      smear_param.restart       = QUDA_BOOLEAN_TRUE;
    // Aniket
    // smear_param.restart = QUDA_BOOLEAN_FALSE;
      gettimeofday ( &ta, (struct timezone *)NULL );
      _performGFlowAdjoint ( point_source_flowed, point_source_flowed, &smear_param, gf_niter, gf_nb, gf_ns );
      gettimeofday ( &tb, (struct timezone *)NULL );
      show_time ( &ta, &tb, "njjn_w_pc_charged_gf_invert_contract", "_performGFlowAdjoint-restart", g_cart_id == 0 );
#endif  // of if _GFLOW_QUDA

    if (isc==1*3+1)  std::cout<<"Haobo: sequential source: "<<point_source_flowed[(((((4*LX+0)*LY+3)*LZ+2)*4+3)*3+1)*2+0]<<std::endl;
    // Haobo: sequential source: -1.35244e-28
    }





  }  /* end of loop on source locations */












  /***************************************************************************/
  /***************************************************************************/

  /***************************************************************************
   * free the allocated memory, finalize
  ***************************************************************************/


#ifndef HAVE_TMLQCD_LIBWRAPPER
  free(g_gauge_field);
#endif
  if ( gauge_field_with_phase != NULL ) free ( gauge_field_with_phase );

  /* free clover matrix terms */
  fini_clover ( &lmzz, &lmzzinv );

  /* free lattice geometry arrays */
  free_geometry();

#ifdef HAVE_TMLQCD_LIBWRAPPER
  tmLQCD_finalise();
#endif


#ifdef HAVE_MPI
  mpi_fini_xchange_contraction();
  mpi_fini_xchange_eo_spinor();
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  gettimeofday ( &end_time, (struct timezone *)NULL );
  show_time ( &start_time, &end_time, "njjn_w_pc_charged_gf_invert_contract", "runtime", g_cart_id == 0 );

  return(0);

}
