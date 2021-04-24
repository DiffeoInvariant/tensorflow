#ifndef _xla_caller_hpp
#define _xla_caller_hpp
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include <tuple>

#include "tensorflow/compiler/xla/array.h"
#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/pjrt/cpu_device.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/tools/hlo_module_loader.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"


using Scalar = double;

xla::Literal from_vec(const std::vector<Scalar>& vals);

xla::Literal from_vec(const std::vector<std::vector<Scalar>>& vals);
  
  
std::pair<std::unique_ptr<xla::PjRtClient>,
	  std::unique_ptr<xla::PjRtExecutable>> build_computation(std::string hlo_module_file, int xla_config_seed=0, bool use_gpu_client = false);



std::shared_ptr<xla::Literal> run_single_output_computation(xla::PjRtExecutable *executable, const std::vector<std::unique_ptr<xla::PjRtBuffer>>& args);

/* a helper function to iterate over the tuple and fill a vector with args; if I == sizeof...(Ts), 
   we're past the end of the tuple, and have nothing left to do */
template<std::size_t I = 0, typename Func_t, typename... Ts>
inline typename std::enable_if<I == sizeof...(Ts), void>::type for_each(const std::tuple<Ts...> &, Func_t) {}

/* the condition for enable_if is I less than sizeof...(Ts). Don't be confused
   by the seemingly-extraneous <, it's there for the less than */
template<std::size_t I = 0, typename Func_t, typename... Ts>
inline typename std::enable_if<I < sizeof...(Ts), void>::type for_each(const std::tuple<Ts...>& t, Func_t f)
{
  f(std::get<I>(t));
  for_each<I + 1, Func_t, Ts...>(t, f);
}

  
/* Containers should be some std::vectors that might contain other nested
   vectors, with Scalar being the bottommost type */
template<typename... Containers>
std::vector<std::unique_ptr<xla::PjRtBuffer>> get_args(const std::tuple<Containers...>& containers, xla::PjRtClient *client, int device_num = 0)
{
  std::vector<std::unique_ptr<xla::PjRtBuffer>> args;
  auto filler = [&args, client, device_num](const auto& vec)
  {
    auto literal = from_vec(vec);
    args.push_back(client->BufferFromHostLiteral(literal, client->addressable_devices()[device_num]).ValueOrDie());
  };
		   
  for_each(containers, filler);
  return args;
}








#endif // _xla_caller_hpp
