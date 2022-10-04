#!/bin/sh
#!/bin/sh
python rl/main_rl.py --env MiniGrid-Combination-Picker-8x8-v0 -n 3000000 -c mask -b target/CAT-0.0.1-SNAPSHOT-jar-with-dependencies.jar --seed 20 --log-dir benchmarks/queries --run-name combpicker8-20-mask-lnet
python rl/main_rl.py --env MiniGrid-Combination-Picker-8x8-v0 -n 3000000 -c none -b target/CAT-0.0.1-SNAPSHOT-jar-with-dependencies.jar --seed 20 --log-dir benchmarks/queries --run-name combpicker8-20-none-lnet


#python rl/main_rl.py --env MiniGrid-Combination-Picker-8x8-v0 -n 3000000 -c mask -b target/CAT-0.0.1-SNAPSHOT-jar-with-dependencies.jar --seed 20 --log-dir benchmarks/queries --run-name combpicker8-20-mask
#python rl/main_rl.py --env MiniGrid-Combination-Picker-8x8-v0 -n 3000000 -c none -b target/CAT-0.0.1-SNAPSHOT-jar-with-dependencies.jar --seed 20 --log-dir benchmarks/queries --run-name combpicker8-20-none
#python rl/main_rl.py --env MiniGrid-Combination-Picker-8x8-v0 -n 3000000 -c mask -b target/CAT-0.0.1-SNAPSHOT-jar-with-dependencies.jar --seed 21 --log-dir benchmarks/queries --run-name combpicker8-21-mask
#python rl/main_rl.py --env MiniGrid-Combination-Picker-8x8-v0 -n 3000000 -c none -b target/CAT-0.0.1-SNAPSHOT-jar-with-dependencies.jar --seed 21 --log-dir benchmarks/queries --run-name combpicker8-21-none
#python rl/main_rl.py --env MiniGrid-Combination-Picker-8x8-v0 -n 3000000 -c mask -b target/CAT-0.0.1-SNAPSHOT-jar-with-dependencies.jar --seed 22 --log-dir benchmarks/queries --run-name combpicker8-22-mask
#python rl/main_rl.py --env MiniGrid-Combination-Picker-8x8-v0 -n 3000000 -c none -b target/CAT-0.0.1-SNAPSHOT-jar-with-dependencies.jar --seed 22 --log-dir benchmarks/queries --run-name combpicker8-22-none
#
#
#python rl/main_rl.py --env MiniGrid-Combination-Picker-Random-16x16-v0 -n 3000000 -c mask -b target/CAT-0.0.1-SNAPSHOT-jar-with-dependencies.jar --seed 20 --log-dir benchmarks/queries --run-name combpicker16-20-mask
#python rl/main_rl.py --env MiniGrid-Combination-Picker-Random-16x16-v0 -n 3000000 -c none -b target/CAT-0.0.1-SNAPSHOT-jar-with-dependencies.jar --seed 20 --log-dir benchmarks/queries --run-name combpicker16-20-none
#python rl/main_rl.py --env MiniGrid-Combination-Picker-Random-16x16-v0 -n 3000000 -c mask -b target/CAT-0.0.1-SNAPSHOT-jar-with-dependencies.jar --seed 21 --log-dir benchmarks/queries --run-name combpicker16-21-mask
#python rl/main_rl.py --env MiniGrid-Combination-Picker-Random-16x16-v0 -n 3000000 -c none -b target/CAT-0.0.1-SNAPSHOT-jar-with-dependencies.jar --seed 21 --log-dir benchmarks/queries --run-name combpicker16-21-none
#python rl/main_rl.py --env MiniGrid-Combination-Picker-Random-16x16-v0 -n 3000000 -c mask -b target/CAT-0.0.1-SNAPSHOT-jar-with-dependencies.jar --seed 22 --log-dir benchmarks/queries --run-name combpicker16-22-mask
#python rl/main_rl.py --env MiniGrid-Combination-Picker-Random-16x16-v0 -n 3000000 -c none -b target/CAT-0.0.1-SNAPSHOT-jar-with-dependencies.jar --seed 22 --log-dir benchmarks/queries --run-name combpicker16-22-none
