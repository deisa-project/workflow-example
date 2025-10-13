/*
================================================================================
 Gray–Scott Reaction–Diffusion (2D) Simulation
================================================================================
A fully distributed Gray–Scott model implemented with:

  • **MPI** for domain decomposition and halo exchange
  • **Kokkos** for portable CPU/GPU parallelism
  • **PyBind11** for integration with the Python-based Doreisa framework

The implementation mirrors the logic of the Python mpi4py version:
  - same halo exchange ordering (U then V)
  - same update rule and seeding pattern
  - same optional visualization output

--------------------------------------------------------------------------------
 Build Example (CMake)
--------------------------------------------------------------------------------
  cmake \
    -DCMAKE_CXX_STANDARD=20 \
    -DKokkos_ENABLE_OPENMP=ON \
    -DKokkos_ENABLE_CUDA=ON \
    -DKokkos_ARCH_{YOUR_ARCH}=ON
  make

--------------------------------------------------------------------------------
 Run Example
--------------------------------------------------------------------------------
  mpirun -n 4 sim_kokkos \
    --steps 2000 --print-every 50 \
    --seed-mode local --periodic \
    --viz-every 50 --viz-gif

================================================================================
*/

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <assert.h>
#include <errno.h>
#include <iostream>
#include <limits.h>
#include <math.h>
#include <mpi.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

// -----------------------------------------------
// Utility: directory creation (no error if exists)
// -----------------------------------------------
static void ensure_dir(const char *path) {
  if (mkdir(path, 0777) != 0 && errno != EEXIST) {
    fprintf(stderr, "[warn] mkdir %s failed: %s\n", path, strerror(errno));
  }
}

// -----------------------------------------------
// Simple RNG (xorshift64*) for portability
// -----------------------------------------------
typedef struct {
  unsigned long long s;
} rng64;

static void rng_seed(rng64 *r, unsigned long long seed) {
  r->s = seed ? seed : 88172645463393265ull;
}
static inline unsigned long long xorshift64s(rng64 *r) {
  unsigned long long x = r->s;
  x ^= x >> 12;
  x ^= x << 25;
  x ^= x >> 27;
  r->s = x;
  return x * 2685821657736338717ull;
}

static inline double rng_uniform01(rng64 *r) {
  return (xorshift64s(r) >> 11) * (1.0 / 9007199254740992.0);
}

// -----------------------------------------------
// Config / CLI
// -----------------------------------------------
struct Config {
  int nx_local;
  int ny_local;
  int px;
  int py;
  int periodic;
  int steps;
  double dt, Du, Dv, F, k;
  int print_every;
  int seed;
  enum { SEED_LOCAL, SEED_GLOBAL } seed_mode;
  double seed_lo, seed_hi;
  int viz_every;
  enum { VIZ_U, VIZ_V, VIZ_BOTH } viz_field;
  char viz_outdir[256];
  double viz_vmin, viz_vmax;
  int viz_gif;
};
static void set_defaults(Config *c) {
  c->nx_local = 256;
  c->ny_local = 256;
  c->px = 0;
  c->py = 0;
  c->periodic = 0;
  c->steps = 500;
  c->dt = 1.0;
  c->Du = 0.16;
  c->Dv = 0.08;
  c->F = 0.060;
  c->k = 0.062;
  c->print_every = 50;
  c->seed = 0;
  c->seed_mode = Config::SEED_LOCAL;
  c->seed_lo = 0.4;
  c->seed_hi = 0.6;
  c->viz_every = 0;
  c->viz_field = Config::VIZ_V;
  strcpy(c->viz_outdir, "frames");
  c->viz_vmin = 0.0;
  c->viz_vmax = 1.0;
  c->viz_gif = 0;
}

static int streq(const char *a, const char *b) { return strcmp(a, b) == 0; }

// Parse simple CLI arguments (only rank 0 prints help)
static void parse_args(Config *c, int argc, char **argv, int rank) {
  set_defaults(c);
  for (int i = 1; i < argc; ++i) {
    if (streq(argv[i], "--nx_local") && i + 1 < argc)
      c->nx_local = atoi(argv[++i]);
    else if (streq(argv[i], "--ny_local") && i + 1 < argc)
      c->ny_local = atoi(argv[++i]);
    else if (streq(argv[i], "--px") && i + 1 < argc)
      c->px = atoi(argv[++i]);
    else if (streq(argv[i], "--py") && i + 1 < argc)
      c->py = atoi(argv[++i]);
    else if (streq(argv[i], "--periodic"))
      c->periodic = 1;
    else if (streq(argv[i], "--steps") && i + 1 < argc)
      c->steps = atoi(argv[++i]);
    else if (streq(argv[i], "--dt") && i + 1 < argc)
      c->dt = atof(argv[++i]);
    else if (streq(argv[i], "--Du") && i + 1 < argc)
      c->Du = atof(argv[++i]);
    else if (streq(argv[i], "--Dv") && i + 1 < argc)
      c->Dv = atof(argv[++i]);
    else if (streq(argv[i], "--F") && i + 1 < argc)
      c->F = atof(argv[++i]);
    else if (streq(argv[i], "--k") && i + 1 < argc)
      c->k = atof(argv[++i]);
    else if (streq(argv[i], "--print-every") && i + 1 < argc)
      c->print_every = atoi(argv[++i]);
    else if (streq(argv[i], "--seed") && i + 1 < argc)
      c->seed = atoi(argv[++i]);
    else if (streq(argv[i], "--seed-mode") && i + 1 < argc) {
      const char *s = argv[++i];
      if (streq(s, "local"))
        c->seed_mode = Config::SEED_LOCAL;
      else if (streq(s, "global"))
        c->seed_mode = Config::SEED_GLOBAL;
    } else if (streq(argv[i], "--seed-frac") && i + 2 < argc) {
      c->seed_lo = atof(argv[++i]);
      c->seed_hi = atof(argv[++i]);
    } else if (streq(argv[i], "--viz-every") && i + 1 < argc)
      c->viz_every = atoi(argv[++i]);
    else if (streq(argv[i], "--viz-field") && i + 1 < argc) {
      const char *s = argv[++i];
      if (streq(s, "U"))
        c->viz_field = Config::VIZ_U;
      else if (streq(s, "V"))
        c->viz_field = Config::VIZ_V;
      else if (streq(s, "both"))
        c->viz_field = Config::VIZ_BOTH;
    } else if (streq(argv[i], "--viz-outdir") && i + 1 < argc) {
      strncpy(c->viz_outdir, argv[++i], sizeof(c->viz_outdir) - 1);
      c->viz_outdir[sizeof(c->viz_outdir) - 1] = '\0';
    } else if (streq(argv[i], "--viz-gif"))
      c->viz_gif = 1;
    else if (streq(argv[i], "-h") || streq(argv[i], "--help")) {
      if (rank == 0) {
        printf("Gray–Scott 2D (MPI + Kokkos)\nOptions:\n  --nx_local INT "
               "[256]\n  --ny_local INT [256]\n  --px INT [auto]\n  --py INT "
               "[auto]\n  --periodic [false]\n  --steps INT [500]\n");
      }
      MPI_Finalize();
      exit(0);
    }
  }
}

// -----------------------------------------------
// Indexing helper for halo layout (row-major, includes ghost cells)
// -----------------------------------------------
#define AT(A, nx, i, j) (A[((i) * ((nx) + 2)) + (j)]) // 0..ny+1, 0..nx+1

// -----------------------------------------------
// Initialization & seeding
// -----------------------------------------------

// Seed a square in local fractional coords (lo,hi) in [0,1]
static void seed_local_fraction(double *U, double *V, int nx, int ny, double lo,
                                double hi) {
  for (int i = 1; i <= ny; ++i)
    for (int j = 1; j <= nx; ++j) {
      double y_norm = ((i - 1) + 0.5) / (double)ny;
      double x_norm = ((j - 1) + 0.5) / (double)nx;
      if (y_norm > lo && y_norm < hi && x_norm > lo && x_norm < hi) {
        AT(U, nx, i, j) = 0.50;
        AT(V, nx, i, j) = 0.25;
      }
    }
}

// Seed a square in global fractional coords (lo,hi) in [0,1]
static void seed_global_fraction(double *U, double *V, int nx, int ny, int px,
                                 int py, int rx, int ry, double lo, double hi) {
  int NX = px * nx, NY = py * ny;
  int x0 = rx * nx, y0 = ry * ny;
  for (int i = 1; i <= ny; ++i)
    for (int j = 1; j <= nx; ++j) {
      double xg = (x0 + (j - 0.5)) / (double)NX;
      double yg = (y0 + (i - 0.5)) / (double)NY;
      if (yg > lo && yg < hi && xg > lo && xg < hi) {
        AT(U, nx, i, j) = 0.50;
        AT(V, nx, i, j) = 0.25;
      }
    }
}

// Initialize U=1, V=0 + small noise + seed square
static void initialize_kokkos(double *Udata, double *Vdata, int nx, int ny,
                              int rx, int ry, int px, int py, rng64 *rng,
                              const Config *cfg) {
  int total = (nx + 2) * (ny + 2);
  // fill U=1 V=0
  Kokkos::parallel_for(
      "init_fill",
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, total),
      KOKKOS_LAMBDA(const int idx) {
        Udata[idx] = 1.0;
        Vdata[idx] = 0.0;
      });
  // add small noise in interior
  int N = nx * ny;
  Kokkos::parallel_for(
      "init_noise",
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, N),
      KOKKOS_LAMBDA(const int idx) {
        int li = idx / nx;
        int lj = idx % nx;
        int i = li + 1;
        int j = lj + 1;
        // noise here
        double noise = 0.02 * rng_uniform01(rng);
        AT(Udata, nx, i, j) -= noise;
        AT(Vdata, nx, i, j) += noise;
      });
  // seed square
  if (cfg->seed_mode == Config::SEED_LOCAL)
    seed_local_fraction(Udata, Vdata, nx, ny, cfg->seed_lo, cfg->seed_hi);
  else
    seed_global_fraction(Udata, Vdata, nx, ny, px, py, rx, ry, cfg->seed_lo,
                         cfg->seed_hi);
}

// -----------------------------------------------
// Neumann BC: copy interior edge into ghost *only* on physical boundaries
// -----------------------------------------------
static void apply_neumann_boundary_ghosts(MPI_Comm cart, double *A, int nx,
                                          int ny) {
  int up, down, left, right;
  MPI_Cart_shift(cart, 0, 1, &up, &down);
  MPI_Cart_shift(cart, 1, 1, &left, &right);

  // Top boundary: no UP neighbor
  if (up == MPI_PROC_NULL) {
    for (int j = 1; j <= nx; ++j)
      AT(A, nx, 0, j) = AT(A, nx, 1, j);
    AT(A, nx, 0, 0) = AT(A, nx, 1, 1);
    AT(A, nx, 0, nx + 1) = AT(A, nx, 1, nx);
  }
  // Bottom boundary: no DOWN neighbor
  if (down == MPI_PROC_NULL) {
    for (int j = 1; j <= nx; ++j)
      AT(A, nx, ny + 1, j) = AT(A, nx, ny, j);
    AT(A, nx, ny + 1, 0) = AT(A, nx, ny, 1);
    AT(A, nx, ny + 1, nx + 1) = AT(A, nx, ny, nx);
  }
  // Left boundary
  if (left == MPI_PROC_NULL) {
    for (int i = 1; i <= ny; ++i)
      AT(A, nx, i, 0) = AT(A, nx, i, 1);
    AT(A, nx, 0, 0) = AT(A, nx, 1, 1);
    AT(A, nx, ny + 1, 0) = AT(A, nx, ny, 1);
  }
  // Right boundary
  if (right == MPI_PROC_NULL) {
    for (int i = 1; i <= ny; ++i)
      AT(A, nx, i, nx + 1) = AT(A, nx, i, nx);
    AT(A, nx, 0, nx + 1) = AT(A, nx, 1, nx);
    AT(A, nx, ny + 1, nx + 1) = AT(A, nx, ny, nx);
  }
}

// -----------------------------------------------
// Halo exchange (Sendrecv rows then columns)
// -----------------------------------------------
static void exchange_halo(MPI_Comm cart, double *A, int nx, int ny) {
  int up, down, left, right;
  MPI_Cart_shift(cart, 0, 1, &up, &down);
  MPI_Cart_shift(cart, 1, 1, &left, &right);

  MPI_Status status;

  // send last interior row to DOWN, receive into top ghost from UP
  MPI_Sendrecv(&AT(A, nx, ny, 1), nx, MPI_DOUBLE, down, 101, &AT(A, nx, 0, 1),
               nx, MPI_DOUBLE, up, 101, cart, &status);

  // send first interior row to UP, receive bottom ghost from DOWN
  MPI_Sendrecv(&AT(A, nx, 1, 1), nx, MPI_DOUBLE, up, 102, &AT(A, nx, ny + 1, 1),
               nx, MPI_DOUBLE, down, 102, cart, &status);

  // Columns: pack/unpack contiguous buffers of length ny
  double *send = (double *)malloc(sizeof(double) * ny);
  double *recv = (double *)malloc(sizeof(double) * ny);

  // pack last interior column
  for (int i = 0; i < ny; ++i)
    send[i] = AT(A, nx, i + 1, nx);

  // send last interior column to RIGHT, receive into first ghost from LEFT
  MPI_Sendrecv(send, ny, MPI_DOUBLE, right, 201, recv, ny, MPI_DOUBLE, left,
               201, cart, &status);

  // unpack column and copy to first ghost column
  for (int i = 0; i < ny; ++i)
    AT(A, nx, i + 1, 0) = recv[i];

  // pack first interior column
  for (int i = 0; i < ny; ++i)
    send[i] = AT(A, nx, i + 1, 1);

  // send first interior column to LEFT, receive into last ghost from RIGHT
  MPI_Sendrecv(send, ny, MPI_DOUBLE, left, 202, recv, ny, MPI_DOUBLE, right,
               202, cart, &status);

  // unpack column and copy to last ghost column
  for (int i = 0; i < ny; ++i)
    AT(A, nx, i + 1, nx + 1) = recv[i];

  free(send);
  free(recv);
}
static void update_ghosts(MPI_Comm cart, double *A, int nx, int ny,
                          int periodic) {
  exchange_halo(cart, A, nx, ny);
  if (!periodic)
    apply_neumann_boundary_ghosts(cart, A, nx, ny);
}

// -----------------------------------------------
// Gray–Scott update on interior cells
// -----------------------------------------------
static void step_gray_scott_kokkos(const double *U, const double *V, double *Un,
                                   double *Vn, int nx, int ny, double Du,
                                   double Dv, double F, double k, double dt) {
  int N = nx * ny;
  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, N),
      KOKKOS_LAMBDA(const int idx) {
        int local_i = idx / nx; // 0..ny-1
        int local_j = idx % nx; // 0..nx-1
        int i = local_i + 1;
        int j = local_j + 1;
        double Ui = AT(U, nx, i, j);
        double Vi = AT(V, nx, i, j);
        double Lu = (-4.0 * Ui + AT(U, nx, i, j - 1) + AT(U, nx, i, j + 1) +
                     AT(U, nx, i - 1, j) + AT(U, nx, i + 1, j));
        double Lv = (-4.0 * Vi + AT(V, nx, i, j - 1) + AT(V, nx, i, j + 1) +
                     AT(V, nx, i - 1, j) + AT(V, nx, i + 1, j));
        double uvv = Ui * Vi * Vi;
        AT(Un, nx, i, j) = Ui + dt * (Du * Lu - uvv + F * (1.0 - Ui));
        AT(Vn, nx, i, j) = Vi + dt * (Dv * Lv + uvv - (F + k) * Vi);
      });
}

// -----------------------------------------------
// Gather helpers (contiguous interior tiles) -> rank 0 assembles global field
// -----------------------------------------------
static double *gather_global_field(MPI_Comm cart, const double *A, int nx,
                                   int ny, int px, int py) {
  int rank, size;
  MPI_Comm_rank(cart, &rank);
  MPI_Comm_size(cart, &size);

  // Pack interior into a contiguous tile
  int tile_elems = nx * ny;
  double *tile = (double *)malloc(sizeof(double) * tile_elems);
  for (int i = 0; i < ny; ++i)
    memcpy(tile + i * nx, &AT(A, nx, i + 1, 1), sizeof(double) * nx);

  // gather tiles
  double *all_tiles = NULL;
  if (rank == 0)
    all_tiles = (double *)malloc(sizeof(double) * tile_elems * size);
  MPI_Gather(tile, tile_elems, MPI_DOUBLE, all_tiles, tile_elems, MPI_DOUBLE, 0,
             cart);

  // gather coords (ry, rx) for each rank
  int coords[2];
  MPI_Cart_coords(cart, rank, 2, coords);
  int *all_coords = NULL;
  if (rank == 0)
    all_coords = (int *)malloc(sizeof(int) * 2 * size);
  MPI_Gather(coords, 2, MPI_INT, all_coords, 2, MPI_INT, 0, cart);

  free(tile);

  if (rank != 0)
    return NULL;

  // reorder gathered array into grid layout according to coords
  int NX = px * nx, NY = py * ny;
  double *G = (double *)malloc(sizeof(double) * NX * NY);

  // place tiles
  for (int r = 0; r < size; ++r) {
    int ry = all_coords[2 * r + 0];
    int rx = all_coords[2 * r + 1];
    double *src = all_tiles + r * tile_elems;
    for (int i = 0; i < ny; ++i)
      memcpy(G + (ry * ny + i) * NX + (rx * nx), src + i * nx,
             sizeof(double) * nx);
  }
  free(all_tiles);
  free(all_coords);
  return G; // owned by caller (rank 0)
}

// -----------------------------------------------
// Write PGM (8-bit grayscale) for a 2D scalar field
// -----------------------------------------------
static void write_pgm(const char *fname, const double *G, int NX, int NY,
                      double vmin, double vmax) {

  // autoscale
  vmin = 1e300;
  vmax = -1e300;
  for (int i = 0; i < NX * NY; ++i) {
    if (G[i] < vmin)
      vmin = G[i];
    if (G[i] > vmax)
      vmax = G[i];
  }
  if (vmax <= vmin) {
    vmin = 0.0;
    vmax = 1.0;
  }
  FILE *f = fopen(fname, "wb");
  if (!f) {
    fprintf(stderr, "[viz] failed to open %s\n", fname);
    return;
  }
  fprintf(f, "P5\n%d %d\n255\n", NX, NY);
  // map to 0..255, origin lower-left to mimic imshow(origin="lower")
  for (int i = NY - 1; i >= 0; --i)
    for (int j = 0; j < NX; ++j) {
      double v = G[i * NX + j];
      double t = (v - vmin) / (vmax - vmin);
      if (t < 0)
        t = 0;
      if (t > 1)
        t = 1;
      unsigned char b = (unsigned char)(t * 255.0 + 0.5);
      fwrite(&b, 1, 1, f);
    }
  fclose(f);
}

static void save_frame(int step, MPI_Comm cart, double *U, double *V, int nx,
                       int ny, int px, int py, int which, const char *outdir,
                       double vmin, double vmax) {
  int rank;
  MPI_Comm_rank(cart, &rank);
  ensure_dir(outdir);
  double *Ug = NULL, *Vg = NULL;
  if (which == Config::VIZ_U || which == Config::VIZ_BOTH)
    Ug = gather_global_field(cart, U, nx, ny, px, py);
  if (which == Config::VIZ_V || which == Config::VIZ_BOTH)
    Vg = gather_global_field(cart, V, nx, ny, px, py);
  if (rank != 0) {
    if (Ug)
      free(Ug);
    if (Vg)
      free(Vg);
    return;
  }
  char path[512];
  int NX = px * nx, NY = py * ny;
  if (which == Config::VIZ_U) {
    snprintf(path, sizeof(path), "%s/step_%06d.pgm", outdir, step);
    write_pgm(path, Ug, NX, NY, vmin, vmax);
  } else if (which == Config::VIZ_V) {
    snprintf(path, sizeof(path), "%s/step_%06d.pgm", outdir, step);
    write_pgm(path, Vg, NX, NY, vmin, vmax);
  } else {
    snprintf(path, sizeof(path), "%s/step_%06d_U.pgm", outdir, step);
    write_pgm(path, Ug, NX, NY, vmin, vmax);
    snprintf(path, sizeof(path), "%s/step_%06d_V.pgm", outdir, step);
    write_pgm(path, Vg, NX, NY, vmin, vmax);
  }
  if (Ug)
    free(Ug);
  if (Vg)
    free(Vg);
}

// -----------------------------------------------
// Main
// -----------------------------------------------
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  namespace py = pybind11;
  // printf("Execution space: %s\n",
  // typeid(Kokkos::DefaultExecutionSpace).name());
  //  Kokkos::print_configuration(std::cout);
  {
    MPI_Comm world = MPI_COMM_WORLD;
    int rank, size;
    MPI_Comm_rank(world, &rank);
    MPI_Comm_size(world, &size);

    // ---- Doreisa integration ----
    py::scoped_interpreter guard{};
    py::dict locals;
    locals["rank"] = rank;

    py::module_ sim_node = py::module_::import("doreisa.simulation_node");
    py::object Client = sim_node.attr("Client");
    py::object client_instance = Client();
    py::exec("print(f'[SIM, rank {rank}] connected to doreisa client')",
             py::globals(), locals);
    // -----------------------------

    Config cfg;
    parse_args(&cfg, argc, argv, rank);

    // Cart topology (py x px). If px/py=0, let MPI choose via Dims_create.
    int dims[2] = {cfg.py, cfg.px};
    MPI_Dims_create(size, 2, dims);
    cfg.py = dims[0];
    cfg.px = dims[1];
    int periods[2] = {cfg.periodic, cfg.periodic};
    MPI_Comm cart;
    MPI_Cart_create(world, 2, dims, periods, 1, &cart);
    int coords[2];
    MPI_Cart_coords(cart, rank, 2, coords);
    int ry = coords[0], rx = coords[1];
    int nx = cfg.nx_local, ny = cfg.ny_local;
    int NX = cfg.px * nx, NY = cfg.py * ny;
    size_t total = (size_t)(nx + 2) * (ny + 2);

    // Allocate fields using HostSpace Views (ny+2) x (nx+2)
    Kokkos::View<double *, Kokkos::HostSpace> U("U", total), V("V", total),
        Un("Un", total), Vn("Vn", total);

    rng64 rng;
    rng_seed(&rng, (unsigned long long)(cfg.seed + rank));
    initialize_kokkos(U.data(), V.data(), nx, ny, rx, ry, cfg.px, cfg.py, &rng,
                      &cfg);

    // Initial ghost update + optional frame 0
    update_ghosts(cart, U.data(), nx, ny, cfg.periodic);
    update_ghosts(cart, V.data(), nx, ny, cfg.periodic);
    if (cfg.viz_every > 0)
      save_frame(0, cart, U.data(), V.data(), nx, ny, cfg.px, cfg.py,
                 cfg.viz_field, cfg.viz_outdir, cfg.viz_vmin, cfg.viz_vmax);

    double t0 = MPI_Wtime();
    for (int step = 0; step < cfg.steps; ++step) {
      double halo_start = MPI_Wtime();
      update_ghosts(cart, U.data(), nx, ny, cfg.periodic);
      update_ghosts(cart, V.data(), nx, ny, cfg.periodic);
      double halo_end = MPI_Wtime();

      double gss_start = MPI_Wtime();
      step_gray_scott_kokkos(U.data(), V.data(), Un.data(), Vn.data(), nx, ny,
                             cfg.Du, cfg.Dv, cfg.F, cfg.k, cfg.dt);
      double gss_end = MPI_Wtime();

      // swap Kokkos views
      std::swap(U, Un);
      std::swap(V, Vn);

      // ---- Doreisa integration ----

      py::tuple coords_tuple = py::make_tuple(coords[0], coords[1]);

      py::array_t<double> U_py({ny + 2, nx + 2}, U.data());
      py::array_t<double> V_py({ny + 2, nx + 2}, V.data());

      py::object Uslice = U_py.attr("__getitem__")(
          py::make_tuple(py::slice(1, -1, 1), py::slice(1, -1, 1)));

      py::object Vslice = V_py.attr("__getitem__")(
          py::make_tuple(py::slice(1, -1, 1), py::slice(1, -1, 1)));

      client_instance.attr("add_chunk")(
          "U", coords_tuple, py::make_tuple(cfg.py, cfg.px), size, step, Uslice,
          py::arg("store_externally") = false);

      client_instance.attr("add_chunk")(
          "V", coords_tuple, py::make_tuple(cfg.py, cfg.px), size, step, Vslice,
          py::arg("store_externally") = false);

      // -----------------------------

      if (cfg.viz_every > 0 && (step % cfg.viz_every == 0))
        save_frame(step, cart, U.data(), V.data(), nx, ny, cfg.px, cfg.py,
                   cfg.viz_field, cfg.viz_outdir, cfg.viz_vmin, cfg.viz_vmax);

      if (cfg.print_every > 0 && (step % cfg.print_every == 0)) {
        MPI_Barrier(cart);
        double local_sum = 0.0;
        for (int i = 1; i <= ny; ++i)
          for (int j = 1; j <= nx; ++j)
            local_sum += AT(U.data(), nx, i, j); // U is current after swap
        double global_sum = 0.0;
        MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, cart);
        if (rank == 0) {
          double elapsed = MPI_Wtime() - t0;
          double gss_ms = (gss_end - gss_start) * 1e3;
          double halo_ms = (halo_end - halo_start) * 1e3;
          printf("[step %6d] ranks=%d grid=%dx%d N=%dx%d local=%dx%d Vsum=%.6e "
                 "elapsed=%.2fs GSS_time=%.2fms halo_time=%.2fms\n",
                 step, size, cfg.py, cfg.px, NY, NX, ny, nx, global_sum,
                 elapsed, gss_ms, halo_ms);
          fflush(stdout);
        }
      }
    }

    MPI_Barrier(cart);
    if (rank == 0) {
      if (cfg.viz_gif && cfg.viz_every > 0) {
        printf("[viz] Frames written to %s (PGM).\n", cfg.viz_outdir);
        printf("[viz] To assemble a GIF: convert %s/step_*.pgm out.gif\n",
               cfg.viz_outdir);
      }
      printf("DONE.\n");
    }

    MPI_Comm_free(&cart);
  }
  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}
