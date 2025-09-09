

envs=(Ant-v4 HalfCheetah-v4 Hopper-v4 Humanoid-v4 Swimmer-v4 Walker2d-v4 BipedalWalker-v3 BipedalWalkerHardcore-v3 LunarLander-v3)
total_steps=(3000000 3000000 1000000 3000000 1000000 3000000 1000000 3000000 1000000)
target_entropies=(-8 -6 -3 -17 -1 -6 -4 -4 -2)
env_kw_params=('{}' '{}' '{}' '{}' '{}' '{}' '{}' '{}' '{"continuous":true,"enable_wind":true,"wind_power":20.0,"turbulence_power":2.0}')
init_beta_params=(0.25 0.0 0.5 0.25 0.0 0.25 0.5 0.0 0.0)


for env_idx in {0..8}
do
	env=${envs[$env_idx]}
	step=${total_steps[$env_idx]}
	target_ent=${target_entropies[$env_idx]}
	env_kw=${env_kw_params[$env_idx]}	
	beta=${init_beta_params[$env_idx]}	
	echo "Environment: ${env}, Target Entropy: ${target_ent}, Total time steps: ${step}, Beta: ${beta}" 
	for seed in {1..5}
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
		--algo_tag "__@delta=0" \
		--max_train_steps=$step \
		--env_kwargs $env_kw

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
		--algo_tag "__@delta_{@pi}=0.01" \
		--max_train_steps=$step \
		--env_kwargs $env_kw

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
		--algo_tag "__@delta_{Q}=0.01" \
		--max_train_steps=$step \
		--env_kwargs $env_kw

		echo "Both Critic and Policy dropout: 0.01"
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
		--algo_tag "__@delta_{@pi, Q}=0.01" \
		--max_train_steps=$step \
		--env_kwargs $env_kw

	done
done
