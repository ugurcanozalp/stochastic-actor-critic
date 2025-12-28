

envs=(RiskyPointMass-v0)
total_steps=(100000)
target_entropies=(-2)
env_kw_params=('{}')

beta_params=("0 0.125 0.25 0.375 0.5")

for env_idx in {0..0}
do
	env=${envs[$env_idx]}
	step=${total_steps[$env_idx]}
	target_ent=${target_entropies[$env_idx]}
	env_kw=${env_kw_params[$env_idx]}	
	betas=${beta_params[$env_idx]}
	echo "Environment: ${env}, Target Entropy: ${target_ent}, Total time steps: ${step}" 
	for beta in $betas 
	do
		echo "Beta: ${beta}"
		for seed in {1..1}
		do 
			echo "Seed: ${seed}"
			python -m scripts.stac \
			--env_name=$env \
			--autotune \
			--alpha 0.01 \
			--target_entropy=$target_ent \
			--beta=$beta \
			--pi_dropout=0.0 \
			--q_dropout=0.0 \
			--seed=$seed \
			--render_mode none \
			--algo_tag "__@beta=${beta}" \
			--max_train_steps=$step \
			--env_kwargs $env_kw
		done
	done
done
