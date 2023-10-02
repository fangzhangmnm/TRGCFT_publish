CHIS=(
    # "8" "24" "30" "40" "44"
    "8" "10" "12" "14" "16" "18" "20" "22" "24" "26" "28" "30" "32" "34" "36" "38" "40" "42" "44"
)

NLAYERS=60

for MAX_DIM in "${CHIS[@]}"
do
    python run_find_critical_temp.py --filename data/hotrg_gilt_chis/X${MAX_DIM}/Tc.pth --log_filename data/hotrg_gilt_chis/X${MAX_DIM}/Tc.txt --nLayers ${NLAYERS} --max_dim ${MAX_DIM} --model Ising2D --param_name beta --param_min 0.440683 --param_max 0.440685 --observable_name magnetization --gilt_enabled --mcf_enabled --tol 1e-15
    python run_HOTRG.py --filename data/hotrg_gilt_chis/X${MAX_DIM}/tensor.pth --nLayers ${NLAYERS} --max_dim ${MAX_DIM} --gilt_enabled --mcf_enabled --model Ising2D --params_file data/hotrg_gilt_chis/X${MAX_DIM}/Tc.pth
    python run_plot_scdims.py --filename data/hotrg_gilt_chis/X${MAX_DIM}/plots --tensor_path data/hotrg_gilt_chis/X${MAX_DIM}/tensor.pth --is_HOTRG --num_scaling_dims 32
done

