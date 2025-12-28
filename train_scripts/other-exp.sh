

envs=(Ant-v4 HalfCheetah-v4 Hopper-v4 Humanoid-v4 Swimmer-v4 Walker2d-v4 BipedalWalker-v3 BipedalWalkerHardcore-v3 LunarLander-v3)
total_steps=(3000000 3000000 1000000 3000000 1000000 3000000 1000000 3000000 1000000)
target_entropies=(-8 -6 -3 -17 -1 -6 -4 -4 -2)
env_kw_params=('{}' '{}' '{}' '{}' '{}' '{}' '{}' '{}' '{"continuous":true,"enable_wind":true,"wind_power":20.0,"turbulence_power":2.0}')


# ESTAC
for env_idx in {0..8}
do
	env=${envs[env_idx]}
	step=${total_steps[env_idx]}
	target_ent=${target_entropies[env_idx]}
	env_kw=${env_kw_params[env_idx]}	
	echo "Environment: ${env}, Target Entropy: ${target_ent}, Total time steps: ${step}" 
	for seed in {1..5}
	do 
		echo "Seed: ${seed}"
		python -m scripts.estac \
		--env_name=$env \
		--autotune \
		--target_entropy=$target_ent \
		--seed=$seed \
		--render_mode none \
		--max_train_steps=$step \
		--env_kwargs $env_kw
	done
done

# DSAC
for env_idx in {0..8}
do
	env=${envs[env_idx]}
	step=${total_steps[env_idx]}
	target_ent=${target_entropies[env_idx]}
	env_kw=${env_kw_params[env_idx]}	
	echo "Environment: ${env}, Target Entropy: ${target_ent}, Total time steps: ${step}" 
	for seed in {1..5}
	do 
		echo "Seed: ${seed}"
		python -m scripts.dsac \
		--env_name=$env \
		--autotune \
		--target_entropy=$target_ent \
		--seed=$seed \
		--render_mode none \
		--max_train_steps=$step \
		--env_kwargs $env_kw
	done
done

# SAC 
for env_idx in {0..8}
do
	env=${envs[env_idx]}
	step=${total_steps[env_idx]}
	target_ent=${target_entropies[env_idx]}
	env_kw=${env_kw_params[env_idx]}	
	echo "Environment: ${env}, Target Entropy: ${target_ent}, Total time steps: ${step}" 
	for seed in {1..5}
	do 
		echo "Seed: ${seed}"
		python -m scripts.sac \
		--env_name=$env \
		--autotune \
		--target_entropy=$target_ent \
		--dropout=0 \
		--seed=$seed \
		--render_mode none \
		--max_train_steps=$step \
		--env_kwargs $env_kw
	done
done
