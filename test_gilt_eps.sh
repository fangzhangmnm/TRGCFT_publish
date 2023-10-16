NLAYERS=60
MAX_DIM=24
# DEVICE="cuda:0"
DEVICE="cuda:1"

EPSS=(
    # "7e-7" "7.5e-7" "8e-7" "8.5e-7" "9e-7" "9.5e-7"
    # 1e-7 2e-7 5e-7 5e-6 8e-6
    2e-6 1e-6 8e-7 1e-5 2e-5

)

for GILT_EPS in "${EPSS[@]}"
do
    python run_find_critical_temp.py --filename data/hotrg_gilt_X${MAX_DIM}_gilt_eps/${GILT_EPS}_Tc.pth --log_filename data/hotrg_gilt_X${MAX_DIM}_gilt_eps/${GILT_EPS}_Tc.txt --nLayers ${NLAYERS} --max_dim ${MAX_DIM} --model Ising2D --param_name beta --param_min 0.440680 --param_max 0.440690 --observable_name magnetization --gilt_enabled --gilt_eps ${GILT_EPS} --mcf_enabled --tol 1e-8  --device ${DEVICE} --method obs
    python run_HOTRG.py --filename data/hotrg_gilt_X${MAX_DIM}_gilt_eps/${GILT_EPS}.pth --nLayers ${NLAYERS} --max_dim ${MAX_DIM}  --gilt_enabled --gilt_eps ${GILT_EPS} --mcf_enabled --model Ising2D --params_file data/hotrg_gilt_X${MAX_DIM}_gilt_eps/${GILT_EPS}_Tc.pth --device ${DEVICE}
    python run_plot_scdims.py --filename data/hotrg_gilt_X${MAX_DIM}_gilt_eps/${GILT_EPS} --tensor_path data/hotrg_gilt_X${MAX_DIM}_gilt_eps/${GILT_EPS}.pth --is_HOTRG --num_scaling_dims 32 --device ${DEVICE}
    # python run_HOTRG.py --filename data/hotrg_gilt_X${MAX_DIM}_gilt_eps1/${GILT_EPS}.pth --nLayers ${NLAYERS} --max_dim ${MAX_DIM}  --gilt_enabled --gilt_eps ${GILT_EPS} --mcf_enabled --model Ising2D --params '{"beta":0.44068381958007813}' --device ${DEVICE}
    # python run_plot_scdims.py --filename data/hotrg_gilt_X${MAX_DIM}_gilt_eps1/${GILT_EPS} --tensor_path data/hotrg_gilt_X${MAX_DIM}_gilt_eps1/${GILT_EPS}.pth --is_HOTRG --num_scaling_dims 32 --device ${DEVICE}
done

