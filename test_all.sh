# Ising2D analytical critical point 0.44068679350977147 gilt_enabled

NLAYERS=60
MAX_DIM=24
DEVICE="cuda:0"
GILT_EPS=8e-7
# beta_critical=0.44068381958007813

# find the critical point, with GILT
python run_find_critical_temp.py --filename data/hotrg_gilt_X${MAX_DIM}_Tc.pth --nLayers ${NLAYERS} --max_dim ${MAX_DIM} --model Ising2D --param_name beta --param_min 0.440680 --param_max 0.440690 --observable_name magnetization  --gilt_enabled --gilt_eps ${GILT_EPS} --mcf_enabled --tol 1e-8 --device ${DEVICE} --method obs --log_filename data/hotrg_gilt_X${MAX_DIM}_Tc.txt
# 0.44068381309509275
# 0.440683818359375 new

# flow at critical point, with GILT
python run_HOTRG.py --filename data/hotrg_gilt_X${MAX_DIM}.pth --nLayers ${NLAYERS} --max_dim ${MAX_DIM}  --gilt_enabled --gilt_eps ${GILT_EPS} --mcf_enabled --model Ising2D --params_file data/hotrg_gilt_X${MAX_DIM}_Tc.pth --device ${DEVICE} 
python run_plot_scdims.py --filename data/hotrg_gilt_X${MAX_DIM} --tensor_path data/hotrg_gilt_X${MAX_DIM}.pth --is_HOTRG --num_scaling_dims 32 --device ${DEVICE} 

# flow at low and high beta, with GILT
python run_HOTRG.py --filename data/hotrg_gilt_X${MAX_DIM}_lowB.pth --nLayers ${NLAYERS} --max_dim ${MAX_DIM}  --gilt_enabled --gilt_eps ${GILT_EPS} --mcf_enabled --model Ising2D --params '{"beta":0.440674}' --device ${DEVICE}
python run_plot_scdims.py --filename data/hotrg_gilt_X${MAX_DIM}_lowB --tensor_path data/hotrg_gilt_X${MAX_DIM}_lowB.pth --is_HOTRG --num_scaling_dims 32 --device ${DEVICE}
python run_HOTRG.py --filename data/hotrg_gilt_X${MAX_DIM}_highB.pth --nLayers ${NLAYERS} --max_dim ${MAX_DIM}  --gilt_enabled --gilt_eps ${GILT_EPS} --mcf_enabled --model Ising2D --params '{"beta":0.440694}' --device ${DEVICE}
python run_plot_scdims.py --filename data/hotrg_gilt_X${MAX_DIM}_highB --tensor_path data/hotrg_gilt_X${MAX_DIM}_highB.pth --is_HOTRG --num_scaling_dims 32 --device ${DEVICE}

# without GILT
python run_find_critical_temp.py --filename data/hotrg_X${MAX_DIM}_Tc.pth --nLayers ${NLAYERS} --max_dim ${MAX_DIM} --model Ising2D --param_name beta --param_min 0.440680 --param_max 0.440700 --observable_name magnetization --mcf_enabled --tol 1e-8 --device ${DEVICE} --method obs --log_filename data/hotrg_X${MAX_DIM}_Tc.txt
# 0.44069609642028806
# 0.44069610351562505 new

python run_HOTRG.py --filename data/hotrg_X${MAX_DIM}.pth --nLayers ${NLAYERS} --max_dim ${MAX_DIM}  --mcf_enabled --model Ising2D --params_file data/hotrg_X${MAX_DIM}_Tc.pth --device ${DEVICE}
python run_plot_scdims.py --filename data/hotrg_X${MAX_DIM} --tensor_path data/hotrg_X${MAX_DIM}.pth --is_HOTRG --num_scaling_dims 32 --device ${DEVICE}

# with GILT, without MCF
python run_HOTRG.py --filename data/hotrg_gilt_X${MAX_DIM}_noMCF.pth --nLayers ${NLAYERS} --max_dim ${MAX_DIM}  --gilt_enabled --gilt_eps ${GILT_EPS} --model Ising2D --params_file data/hotrg_gilt_X${MAX_DIM}_Tc.pth --device ${DEVICE}
python run_plot_scdims.py --filename data/hotrg_gilt_X${MAX_DIM}_noMCF --tensor_path data/hotrg_gilt_X${MAX_DIM}_noMCF.pth --is_HOTRG --num_scaling_dims 32 --device ${DEVICE}

# without GILT, without MCF
python run_HOTRG.py --filename data/hotrg_X${MAX_DIM}_noMCF.pth --nLayers ${NLAYERS} --max_dim ${MAX_DIM} --model Ising2D --params_file data/hotrg_X${MAX_DIM}_Tc.pth --device ${DEVICE}
python run_plot_scdims.py --filename data/hotrg_X${MAX_DIM}_noMCF --tensor_path data/hotrg_X${MAX_DIM}_noMCF.pth --is_HOTRG --num_scaling_dims 32 --device ${DEVICE}


# n-point functions
python run_gen_correlation_points.py

# critical, lowB, highB
python run_calculate_correlation.py --filename data/hotrg_gilt_X24_correlation.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/hotrg_gilt_X24.pth --log2Size 30 --device ${DEVICE}
python run_calculate_correlation.py --filename data/hotrg_gilt_X24_lowB_correlation.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/hotrg_gilt_X24_lowB.pth --log2Size 30 --device ${DEVICE}
python run_calculate_correlation.py --filename data/hotrg_gilt_X24_highB_correlation.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/hotrg_gilt_X24_highB.pth --log2Size 30 --device ${DEVICE}

python run_calculate_observables.py --output_filename data/hotrg_gilt_X24_observables.csv --tensors_filename data/hotrg_gilt_X24.pth --observables magnetization --double_layer --device ${DEVICE}
python run_calculate_observables.py --output_filename data/hotrg_gilt_X24_lowB_observables.csv --tensors_filename data/hotrg_gilt_X24_lowB.pth --observables magnetization --double_layer --device ${DEVICE}
python run_calculate_observables.py --output_filename data/hotrg_gilt_X24_highB_observables.csv --tensors_filename data/hotrg_gilt_X24_highB.pth --observables magnetization --double_layer --device ${DEVICE}


# nogilt, 00, nogilt 00
python run_calculate_correlation.py --filename data/hotrg_X24_correlation.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/hotrg_X24.pth --log2Size 30 --device ${DEVICE}
python run_calculate_correlation.py --filename data/hotrg_gilt_X24_correlation_00.pkl --points_filename data/2pt_correlation_points_30_00.pkl --tensors_filename data/hotrg_gilt_X24.pth --log2Size 30 --device ${DEVICE}
python run_calculate_correlation.py --filename data/hotrg_X24_correlation_00.pkl --points_filename data/2pt_correlation_points_30_00.pkl --tensors_filename data/hotrg_X24.pth --log2Size 30 --device ${DEVICE}

# additional points for improved precision
python run_calculate_correlation.py --filename data/hotrg_gilt_X24_correlation_appended.pkl --points_filename data/2pt_correlation_points_30_appended.pkl --tensors_filename data/hotrg_gilt_X24.pth --log2Size 30 --device ${DEVICE}
python run_calculate_correlation.py --filename data/hotrg_gilt_X24_lowB_correlation_appended.pkl --points_filename data/2pt_correlation_points_30_appended.pkl --tensors_filename data/hotrg_gilt_X24_lowB.pth --log2Size 30 --device ${DEVICE}
python run_calculate_correlation.py --filename data/hotrg_gilt_X24_highB_correlation_appended.pkl --points_filename data/2pt_correlation_points_30_appended.pkl --tensors_filename data/hotrg_gilt_X24_highB.pth --log2Size 30 --device ${DEVICE}

# four points
python run_calculate_correlation.py --filename data/hotrg_gilt_X24_4pt_correlation_10.pkl --points_filename data/4pt_correlation_points_10.pkl --tensors_filename data/hotrg_gilt_X24.pth --log2Size 10 --device ${DEVICE}
python run_calculate_correlation.py --filename data/hotrg_gilt_X24_4pt_correlation.pkl --points_filename data/4pt_correlation_points.pkl --tensors_filename data/hotrg_gilt_X24.pth --log2Size 30 --device ${DEVICE}

# smearing corners
python run_calculate_correlation.py --filename data/hotrg_X24_smearing_corner_10.pkl --points_filename data/smearing_corner_10.pkl --tensors_filename data/hotrg_X24.pth --log2Size 10 --device ${DEVICE}
python run_calculate_correlation.py --filename data/hotrg_gilt_X24_smearing_corner_10.pkl --points_filename data/smearing_corner_10.pkl --tensors_filename data/hotrg_gilt_X24.pth --log2Size 10 --device ${DEVICE}

# torus, with and without gilt. scan along y
python run_calculate_correlation.py --filename data/hotrg_X24_torus_correlation_y_10.pkl --points_filename data/torus_correlation_points_y_10.pkl --tensors_filename data/hotrg_X24.pth --log2Size 10 --device ${DEVICE}
python run_calculate_correlation.py --filename data/hotrg_gilt_X24_torus_correlation_y_10.pkl --points_filename data/torus_correlation_points_y_10.pkl --tensors_filename data/hotrg_gilt_X24.pth --log2Size 10 --device ${DEVICE}

# data/torus_correlation_points_y_mid_10.pkl
python run_calculate_correlation.py --filename data/hotrg_X24_torus_correlation_y_mid_10.pkl --points_filename data/torus_correlation_points_y_mid_10.pkl --tensors_filename data/hotrg_X24.pth --log2Size 10 --device ${DEVICE}
python run_calculate_correlation.py --filename data/hotrg_gilt_X24_torus_correlation_y_mid_10.pkl --points_filename data/torus_correlation_points_y_mid_10.pkl --tensors_filename data/hotrg_gilt_X24.pth --log2Size 10 --device ${DEVICE}

# torus big, y=0
# data/torus_correlation_points_30_00.pkl
python run_calculate_correlation.py --filename data/hotrg_X24_torus_correlation_30_00.pkl --points_filename data/torus_correlation_points_30_00.pkl --tensors_filename data/hotrg_X24.pth --log2Size 30 --device ${DEVICE}
python run_calculate_correlation.py --filename data/hotrg_gilt_X24_torus_correlation_30_00.pkl --points_filename data/torus_correlation_points_30_00.pkl --tensors_filename data/hotrg_gilt_X24.pth --log2Size 30 --device ${DEVICE}

# torus big, y=mid
# data/torus_correlation_points_30_mid.pkl
python run_calculate_correlation.py --filename data/hotrg_X24_torus_correlation_30_mid.pkl --points_filename data/torus_correlation_points_30_mid.pkl --tensors_filename data/hotrg_X24.pth --log2Size 30 --device ${DEVICE}
python run_calculate_correlation.py --filename data/hotrg_gilt_X24_torus_correlation_30_mid.pkl --points_filename data/torus_correlation_points_30_mid.pkl --tensors_filename data/hotrg_gilt_X24.pth --log2Size 30 --device ${DEVICE}




