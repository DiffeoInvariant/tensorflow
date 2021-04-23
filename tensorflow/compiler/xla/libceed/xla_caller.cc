#include "xla_caller.hpp"


xla::Literal<Scalar> from_vec(const std::vector<Scalar>& vals)
{
  auto size = static_cast<int64>(vals.size());
  xla::Array<Scalar> arr({size});
  arr.SetValues(vals);
  return xla::Literal<Scalar>::CreateFromArray(arr);
}

xla::Literal<Scalar> from_vec(const std::vector<std::vector<Scalar>>& vals)
{
  auto rows = static_cast<int64>(vals.size());
  auto cols = static_cast<int64>(vals[0].size());
  xla::Array2D<Scalar> arr({rows, cols});
  auto filler = [&vals](int64 row, int64 col, Scalar *val) { *val = vals[row][col]; };
  /* since an Array2D doesn't have a SetValues that takes in a vector of vectors, do the same via Each */
  arr.Each(filler);
  return xla::LiteralUtil::CreateR2FromArray2D(arr);
}


std::pair<std::unique_ptr<xla::PjRtClient>,
	  std::unique_ptr<xla::PjRtExecutable>> build_computation(std::string hlo_module_file, int xla_config_seed, bool use_gpu_client)
{
  std::function<void(xla::HloModuleConfig*)> config_modifier_hook = [](xla::HloModuleConfig* config) { config->set_seed(xla_config_seed); };
 
  std::unique_ptr<xla::HloModule> hlo_module = LoadModuleFromFile(hlo_module_file,
								  xla::hlo_module_loader_details::Config(),
								  "txt",
								  config_modifier_hook).ValueOrDie();
  const xla::HloModuleProto hlo_pb = hlo_module->ToProto();

  std::unique_ptr<xla::PjRtClient> client = (use_gpu_client ? xla::GetGpuClient(/*asynchronous=*/true).ValueOrDie() : xla::GetCpuClient(/*asynchronous=*/true).ValueOrDie());

  xla::XlaComputation computation(hlo_pb);
  xla::CompileOptions compile_opts;
  std::unique_ptr<xla::PjRtExecutable> executable = client->compile(computation, compile_opts).ValueOrDie();

  return {client, executable};
}


std::shared_ptr<xla::Literal> run_single_output_computation(xla::PjRtExecutable *executable, const std::vector<std::unique_ptr<xla::PjRtBuffer>>& args);
{
  xla::ExecuteOptions options;
  return executable->Execute({args}, options).ValueOrDie()[0][0]->ToLiteral().ValueOrDie();
}
								  
