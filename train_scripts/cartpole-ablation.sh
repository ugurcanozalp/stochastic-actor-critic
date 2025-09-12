

envs=(CartPoleSwingUp-v1 CartPoleSwingUp-v1)
total_steps=(50000 50000)
target_entropies=(-1 -1)
env_kw_params=('{"continuous":true,"stochastic":false}' '{"continuous":true,"stochastic":true}')
env_tags=('-Deterministic' '-Stochastic')


# STAC
for env_idx in {0..1}
do
	env=${envs[env_idx]}
	step=${total_steps[env_idx]}
	target_ent=${target_entropies[env_idx]}
	env_tag=${env_tags[env_idx]}		
	env_kw=${env_kw_params[env_idx]}	
	echo "Environment: ${env}, Target Entropy: ${target_ent}, Total time steps: ${step}" 
	for seed in {1..10}
	do 
		echo "Seed: ${seed}"
		python -m scripts.stac \
		--env_name=$env \
		--autotune \
		--target_entropy=$target_ent \
		--autopessimism \
		--beta=0.25 \
		--pi_dropout=0.01 \
		--q_dropout=0.01 \
		--seed=$seed \
		--render_mode none \
		--env_tag=$env_tag \
		--max_train_steps=$step \
		--env_kwargs $env_kw \
		--start_steps 1000 \
		--save_memory	
	done
done


# DSACT
for env_idx in {0..1}
do
	env=${envs[env_idx]}
	step=${total_steps[env_idx]}
	target_ent=${target_entropies[env_idx]}
	env_tag=${env_tags[env_idx]}		
	env_kw=${env_kw_params[env_idx]}	
	echo "Environment: ${env}, Target Entropy: ${target_ent}, Total time steps: ${step}" 
	for seed in {1..10}
	do 
		echo "Seed: ${seed}"
		python -m scripts.dsact \
		--env_name=$env \
		--autotune \
		--target_entropy=$target_ent \
		--seed=$seed \
		--render_mode none \
		--env_tag=$env_tag \
		--max_train_steps=$step \
		--env_kwargs $env_kw \
		--start_steps 1000 \
		--save_memory	
	done
done


# TOPSAC 
for env_idx in {0..1}
do
	env=${envs[env_idx]}
	step=${total_steps[env_idx]}
	target_ent=${target_entropies[env_idx]}
	env_tag=${env_tags[env_idx]}		
	env_kw=${env_kw_params[env_idx]}	
	echo "Environment: ${env}, Target Entropy: ${target_ent}, Total time steps: ${step}" 
	for seed in {1..10}
	do 
		echo "Seed: ${seed}"
		python -m scripts.topsac \
		--env_name=$env \
		--autotune \
		--target_entropy=$target_ent \
		--seed=$seed \
		--render_mode none \
		--env_tag=$env_tag \
		--max_train_steps=$step \
		--env_kwargs $env_kw \
		--start_steps 1000 \
		--save_memory
	done
done

# SAC 
for env_idx in {0..1}
do
	env=${envs[env_idx]}
	step=${total_steps[env_idx]}
	target_ent=${target_entropies[env_idx]}
	env_tag=${env_tags[env_idx]}		
	env_kw=${env_kw_params[env_idx]}	
	echo "Environment: ${env}, Target Entropy: ${target_ent}, Total time steps: ${step}" 
	for seed in {1..10}
	do 
		echo "Seed: ${seed}"
		python -m scripts.sac \
		--env_name=$env \
		--autotune \
		--target_entropy=$target_ent \
		--dropout=0 \
		--seed=$seed \
		--render_mode none \
		--env_tag=$env_tag \
		--max_train_steps=$step \
		--env_kwargs $env_kw \
		--start_steps 1000 \
		--save_memory
	done
done