[Drivers]
    [unit]
      type = ModelUnitTest
      model = 'model'
      input_Scalar_names = 'state/ms'
      input_Scalar_values = 'ms'
      output_Scalar_names = 'state/alpha'
      output_Scalar_values = 'alpha'
      check_AD_parameter_derivatives = false
  []
[]

[Tensors]
    [ms]
        type = Scalar
        values = "0.03 0.75 0.22"
        batch_shape = '(3)'
    []
    [alpha]
        type = Scalar
        values = "4.75 -1.25 3.166666"
        batch_shape = '(3)'
    []
[]

[Models]
    [model]
        type = PyrolysisConversionAmount
        initial_mass_solid = 0.2
        initial_mass_binder = 0.4
        reaction_yield = 0.7

        mass_solid = 'state/ms'
        reaction_amount = 'state/alpha'
    []
[]