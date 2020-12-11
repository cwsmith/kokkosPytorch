#include <Kokkos_Core.hpp>
#include <cstdio>
#include <torch/script.h>
using view_type = Kokkos::View<double * [3]>;

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);

  printf("foo\n");
  torch::jit::script::Module module;
  try {
    printf("before loading pytorch model \"%s\" \n", argv[1]);
    module = torch::jit::load(argv[1]);
    printf("loaded pytorch model \"%s\" \n", argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  {
    view_type a("A", 10);

#if defined(KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA)
    Kokkos::parallel_for(
        10, KOKKOS_LAMBDA(const int i) {
          // Acesss the View just like a Fortran array.  The layout depends
          // on the View's memory space, so don't rely on the View's
          // physical memory layout unless you know what you're doing.
          a(i, 0) = 1.0 * i;
          a(i, 1) = 1.0 * i * i;
          a(i, 2) = 1.0 * i * i * i;
        });
    // Reduction functor that reads the View given to its constructor.
    double sum = 0;
    Kokkos::parallel_reduce(
        10,
        KOKKOS_LAMBDA(const int i, double& lsum) {
          lsum += a(i, 0) * a(i, 1) / (a(i, 2) + 0.1);
        },
        sum);
    printf("Result: %f\n", sum);
#endif
  }
  Kokkos::finalize();
}
