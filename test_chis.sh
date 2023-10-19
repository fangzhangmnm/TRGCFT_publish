CHIS=(
    # "8" "24" "30" "40" "44"
    "8" "10" "12" "14" "16" "18" "20" "22" "24" "26" "28" "30" "32" "34" "36" "38" "40" "42" "44"
)

NLAYERS=60
DEVICE="cuda:1"
GILT_EPS=8e-7

for MAX_DIM in "${CHIS[@]}"
do
    python run_find_critical_temp.py --filename data/hotrg_gilt_chis/X${MAX_DIM}/Tc.pth --log_filename data/hotrg_gilt_chis/X${MAX_DIM}/Tc.txt --nLayers ${NLAYERS} --max_dim ${MAX_DIM} --model Ising2D --param_name beta --param_min 0.440680 --param_max 0.440690 --observable_name magnetization --gilt_enabled --gilt_eps ${GILT_EPS} --mcf_enabled --tol 1e-8 --method obs --device ${DEVICE}
    python run_HOTRG.py --filename data/hotrg_gilt_chis/X${MAX_DIM}/tensor.pth --nLayers ${NLAYERS} --max_dim ${MAX_DIM} --gilt_enabled --gilt_eps ${GILT_EPS} --mcf_enabled --model Ising2D --params_file data/hotrg_gilt_chis/X${MAX_DIM}/Tc.pth --device ${DEVICE}
    python run_plot_scdims.py --filename data/hotrg_gilt_chis/X${MAX_DIM}/plots --tensor_path data/hotrg_gilt_chis/X${MAX_DIM}/tensor.pth --is_HOTRG --num_scaling_dims 32 --device ${DEVICE}
    python run_get_ltrg_eigvecs.py --filename data/hotrg_gilt_chis/X${MAX_DIM}/lTRG_gilt_L30.pth --tensor_path data/hotrg_gilt_chis/X${MAX_DIM}/tensor.pth --iLayer 30 --mcf_enabled --gilt_enabled --gilt_eps ${GILT_EPS}  --svd_num_eigvecs 32 --device ${DEVICE}
done

