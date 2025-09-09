

envs=(CartPoleSwingUp-v1 CartPoleSwingUp-v1)
total_steps=(50000 50000)
target_entropies=(-1 -1)
env_kw_params=('{"continuous":true,"stochastic":false}' '{"continuous":true,"stochastic":true}')
env_tags=('-Deterministic' '-Stochastic')

beta=0.25

for env_idx in {0..1}
do
	env=${envs[$env_idx]}
	step=${total_steps[$env_idx]}
	target_ent=${target_entropies[$env_idx]}
	env_kw=${env_kw_params[$env_idx]}
	env_tag=${env_tags[env_idx]}	
	betas=${beta_params[$env_idx]}
	echo "Environment: ${env}, Target Entropy: ${target_ent}, Total time steps: ${step}, Env Tag: ${env_tag}" 
	for seed in {1..10}
	do 
		echo "Seed: ${seed}"

		echo "No dropout"
		python -m scripts.stac \
		--env_name=$env \
		--autotune \
		--target_entropy=$target_ent \
		--autopessimism \
		--beta=$beta \
		--pi_dropout=0 \
		--q_dropout=0 \
		--seed=$seed \
		--render_mode none \
		--algo_tag "__@beta=${beta}" \
		--env_tag=$env_tag \
		--max_train_steps=$step \
		--env_kwargs $env_kw \
		--start_steps 1000 \
		--save_memory

		echo "Policy dropout: 0.01"
		python -m scripts.stac \
		--env_name=$env \
		--autotune \
		--target_entropy=$target_ent \
		--autopessimism \
		--beta=$beta \
		--pi_dropout=0.01 \
		--q_dropout=0 \
		--seed=$seed \
		--render_mode none \
		--algo_tag "__@beta=${beta}__@delta_{@pi}=0.01" \
		--env_tag=$env_tag \
		--max_train_steps=$step \
		--env_kwargs $env_kw \
		--start_steps 1000 \
		--save_memory

		echo "Critic dropout: 0.01"
		python -m scripts.stac \
		--env_name=$env \
		--autotune \
		--target_entropy=$target_ent \
		--autopessimism \
		--beta=$beta \
		--pi_dropout=0 \
		--q_dropout=0.01 \
		--seed=$seed \
		--render_mode none \
		--algo_tag "__@beta=${beta}__@delta_{Q}=0.01" \
		--env_tag=$env_tag \
		--max_train_steps=$step \
		--env_kwargs $env_kw \
		--start_steps 1000	\
		--save_memory		

		echo "Policy/Critic dropout: 0.01"
		python -m scripts.stac \
		--env_name=$env \
		--autotune \
		--target_entropy=$target_ent \
		--autopessimism \
		--beta=$beta \
		--pi_dropout=0.01 \
		--q_dropout=0.01 \
		--seed=$seed \
		--render_mode none \
		--algo_tag "__@beta=${beta}__@delta_{@pi,Q}=0.01" \
		--env_tag=$env_tag \
		--max_train_steps=$step \
		--env_kwargs $env_kw \
		--start_steps 1000	\
		--save_memory		

	done
done
