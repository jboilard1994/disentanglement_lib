set workdir="%cd%"
echo %workdir%
python benchmark_launch_scripts/dev/start_modcompact_benchmark.py --cwd %workdir%
python benchmark_launch_scripts/dev/start_noise_benchmark.py --cwd %workdir%
python benchmark_launch_scripts/dev/start_nonlinear_benchmark.py --cwd %workdir%
python benchmark_launch_scripts/dev/start_rotation_benchmark.py --cwd %workdir%

