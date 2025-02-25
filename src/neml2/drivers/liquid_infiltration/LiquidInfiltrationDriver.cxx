// Copyright 2024, UChicago Argonne, LLC
// All Rights Reserved
// Software Name: NEML2 -- the New Engineering material Model Library, version 2
// By: Argonne National Laboratory
// OPEN SOURCE LICENSE (MIT)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "neml2/drivers/liquid_infiltration/LiquidInfiltrationDriver.h"

namespace neml2
{
register_NEML2_object(LiquidInfiltrationDriver);

OptionSet
LiquidInfiltrationDriver::expected_options()
{
  OptionSet options = TransientDriver::expected_options();

  options.set<CrossRef<torch::Tensor>>("Prescribed_Liquid_Inlet_Rate");
  options.set<VariableName>("Liquid_Inlet_Rate") = VariableName("forces", "aInDot");
  return options;
}

LiquidInfiltrationDriver::LiquidInfiltrationDriver(const OptionSet & options)
  : TransientDriver(options),
    _driving_force(options.get<CrossRef<torch::Tensor>>("Prescribed_Liquid_Inlet_Rate"), 2),
    _driving_force_name(options.get<VariableName>("Liquid_Inlet_Rate"))
{
  _driving_force = _driving_force.to(_device);
}

void
LiquidInfiltrationDriver::diagnose(std::vector<Diagnosis> & diagnoses) const
{
  TransientDriver::diagnose(diagnoses);
}

void
LiquidInfiltrationDriver::update_forces()
{
  TransientDriver::update_forces();

  auto current_driving_force = _driving_force.batch_index({_step_count});
  _in.base_index_put_(_driving_force_name, current_driving_force);
}
}
