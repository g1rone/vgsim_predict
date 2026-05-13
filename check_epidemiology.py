import inspect
import VGsim


simulator = VGsim.Simulator(1, 2, 1, seed=42)

print("signature output_epidemiology_timelines:")
print(inspect.signature(simulator.output_epidemiology_timelines))

print("\nsource output_epidemiology_timelines:")
print(inspect.getsource(simulator.output_epidemiology_timelines))