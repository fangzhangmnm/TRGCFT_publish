# Ising2D analytical critical point 0.44068679350977147

NLAYERS=60
MAX_DIM=24
# beta_critical=0.44068381958007813

# find the critical point, with GILT
python run_find_critical_temp.py --filename data/hotrg_gilt_X${MAX_DIM}_Tc.pth --nLayers ${NLAYERS} --max_dim ${MAX_DIM} --model Ising2D --param_name beta --param_min 0.4406 --param_max 0.4407 --observable_name magnetization --gilt_enabled --mcf_enabled

# flow at critical point, with GILT
python run_HOTRG.py --filename data/hotrg_gilt_X${MAX_DIM}.pth --nLayers ${NLAYERS} --max_dim ${MAX_DIM} --gilt_enabled --mcf_enabled --model Ising2D --params_file data/hotrg_gilt_X${MAX_DIM}_Tc.pth
python run_plot_scdims.py --filename data/hotrg_gilt_X${MAX_DIM} --tensor_path data/hotrg_gilt_X${MAX_DIM}.pth --is_HOTRG --num_scaling_dims 32

# flow at low and high beta, with GILT
python run_HOTRG.py --filename data/hotrg_gilt_X${MAX_DIM}_lowB.pth --nLayers ${NLAYERS} --max_dim ${MAX_DIM} --gilt_enabled --mcf_enabled --model Ising2D --params '{"beta":0.440674}'
python run_plot_scdims.py --filename data/hotrg_gilt_X${MAX_DIM}_lowB --tensor_path data/hotrg_gilt_X${MAX_DIM}_lowB.pth --is_HOTRG --num_scaling_dims 32
python run_HOTRG.py --filename data/hotrg_gilt_X${MAX_DIM}_highB.pth --nLayers ${NLAYERS} --max_dim ${MAX_DIM} --gilt_enabled --mcf_enabled --model Ising2D --params '{"beta":0.440694}'
python run_plot_scdims.py --filename data/hotrg_gilt_X${MAX_DIM}_highB --tensor_path data/hotrg_gilt_X${MAX_DIM}_highB.pth --is_HOTRG --num_scaling_dims 32

# without GILT
python run_find_critical_temp.py --filename data/hotrg_X${MAX_DIM}_Tc.pth --nLayers ${NLAYERS} --max_dim ${MAX_DIM} --model Ising2D --param_name beta --param_min 0.4406 --param_max 0.4407 --observable_name magnetization --mcf_enabled
python run_HOTRG.py --filename data/hotrg_X${MAX_DIM}.pth --nLayers ${NLAYERS} --max_dim ${MAX_DIM}  --mcf_enabled --model Ising2D --params_file data/hotrg_X${MAX_DIM}_Tc.pth
python run_plot_scdims.py --filename data/hotrg_X${MAX_DIM} --tensor_path data/hotrg_X${MAX_DIM}.pth --is_HOTRG --num_scaling_dims 32

# with GILT, without MCF
python run_HOTRG.py --filename data/hotrg_gilt_X${MAX_DIM}_noMCF.pth --nLayers ${NLAYERS} --max_dim ${MAX_DIM} --gilt_enabled --model Ising2D --params_file data/hotrg_gilt_X${MAX_DIM}_Tc.pth
python run_plot_scdims.py --filename data/hotrg_gilt_X${MAX_DIM}_noMCF --tensor_path data/hotrg_gilt_X${MAX_DIM}_noMCF.pth --is_HOTRG --num_scaling_dims 32

# without GILT, without MCF
python run_HOTRG.py --filename data/hotrg_X${MAX_DIM}_noMCF.pth --nLayers ${NLAYERS} --max_dim ${MAX_DIM} --model Ising2D --params_file data/hotrg_X${MAX_DIM}_Tc.pth
python run_plot_scdims.py --filename data/hotrg_X${MAX_DIM}_noMCF --tensor_path data/hotrg_X${MAX_DIM}_noMCF.pth --is_HOTRG --num_scaling_dims 32

# eigenvalue of layers
python run_get_ltrg_eigvecs.py --filename data/layers/hotrg_gilt_X24_lTRG_gilt_L2.pth --tensor_path data/hotrg_gilt_X24.pth --iLayer 2 --mcf_enabled --gilt_enabled  --svd_num_eigvecs 32
python run_get_ltrg_eigvecs.py --filename data/layers/hotrg_gilt_X24_lTRG_gilt_L4.pth --tensor_path data/hotrg_gilt_X24.pth --iLayer 4 --mcf_enabled --gilt_enabled  --svd_num_eigvecs 32
python run_get_ltrg_eigvecs.py --filename data/layers/hotrg_gilt_X24_lTRG_gilt_L6.pth --tensor_path data/hotrg_gilt_X24.pth --iLayer 6 --mcf_enabled --gilt_enabled  --svd_num_eigvecs 32
python run_get_ltrg_eigvecs.py --filename data/layers/hotrg_gilt_X24_lTRG_gilt_L8.pth --tensor_path data/hotrg_gilt_X24.pth --iLayer 8 --mcf_enabled --gilt_enabled  --svd_num_eigvecs 32
python run_get_ltrg_eigvecs.py --filename data/layers/hotrg_gilt_X24_lTRG_gilt_L10.pth --tensor_path data/hotrg_gilt_X24.pth --iLayer 10 --mcf_enabled --gilt_enabled  --svd_num_eigvecs 32
python run_get_ltrg_eigvecs.py --filename data/layers/hotrg_gilt_X24_lTRG_gilt_L12.pth --tensor_path data/hotrg_gilt_X24.pth --iLayer 12 --mcf_enabled --gilt_enabled  --svd_num_eigvecs 32
python run_get_ltrg_eigvecs.py --filename data/layers/hotrg_gilt_X24_lTRG_gilt_L14.pth --tensor_path data/hotrg_gilt_X24.pth --iLayer 14 --mcf_enabled --gilt_enabled  --svd_num_eigvecs 32
python run_get_ltrg_eigvecs.py --filename data/layers/hotrg_gilt_X24_lTRG_gilt_L16.pth --tensor_path data/hotrg_gilt_X24.pth --iLayer 16 --mcf_enabled --gilt_enabled  --svd_num_eigvecs 32
python run_get_ltrg_eigvecs.py --filename data/layers/hotrg_gilt_X24_lTRG_gilt_L18.pth --tensor_path data/hotrg_gilt_X24.pth --iLayer 18 --mcf_enabled --gilt_enabled  --svd_num_eigvecs 32
python run_get_ltrg_eigvecs.py --filename data/layers/hotrg_gilt_X24_lTRG_gilt_L20.pth --tensor_path data/hotrg_gilt_X24.pth --iLayer 20 --mcf_enabled --gilt_enabled  --svd_num_eigvecs 32
python run_get_ltrg_eigvecs.py --filename data/layers/hotrg_gilt_X24_lTRG_gilt_L22.pth --tensor_path data/hotrg_gilt_X24.pth --iLayer 22 --mcf_enabled --gilt_enabled  --svd_num_eigvecs 32
python run_get_ltrg_eigvecs.py --filename data/layers/hotrg_gilt_X24_lTRG_gilt_L24.pth --tensor_path data/hotrg_gilt_X24.pth --iLayer 24 --mcf_enabled --gilt_enabled  --svd_num_eigvecs 32
python run_get_ltrg_eigvecs.py --filename data/layers/hotrg_gilt_X24_lTRG_gilt_L26.pth --tensor_path data/hotrg_gilt_X24.pth --iLayer 26 --mcf_enabled --gilt_enabled  --svd_num_eigvecs 32
python run_get_ltrg_eigvecs.py --filename data/layers/hotrg_gilt_X24_lTRG_gilt_L28.pth --tensor_path data/hotrg_gilt_X24.pth --iLayer 28 --mcf_enabled --gilt_enabled  --svd_num_eigvecs 32
python run_get_ltrg_eigvecs.py --filename data/layers/hotrg_gilt_X24_lTRG_gilt_L30.pth --tensor_path data/hotrg_gilt_X24.pth --iLayer 30 --mcf_enabled --gilt_enabled  --svd_num_eigvecs 32
python run_get_ltrg_eigvecs.py --filename data/layers/hotrg_gilt_X24_lTRG_gilt_L32.pth --tensor_path data/hotrg_gilt_X24.pth --iLayer 32 --mcf_enabled --gilt_enabled  --svd_num_eigvecs 32
python run_get_ltrg_eigvecs.py --filename data/layers/hotrg_gilt_X24_lTRG_gilt_L34.pth --tensor_path data/hotrg_gilt_X24.pth --iLayer 34 --mcf_enabled --gilt_enabled  --svd_num_eigvecs 32
python run_get_ltrg_eigvecs.py --filename data/layers/hotrg_gilt_X24_lTRG_gilt_L36.pth --tensor_path data/hotrg_gilt_X24.pth --iLayer 36 --mcf_enabled --gilt_enabled  --svd_num_eigvecs 32
python run_get_ltrg_eigvecs.py --filename data/layers/hotrg_gilt_X24_lTRG_gilt_L38.pth --tensor_path data/hotrg_gilt_X24.pth --iLayer 38 --mcf_enabled --gilt_enabled  --svd_num_eigvecs 32
python run_get_ltrg_eigvecs.py --filename data/layers/hotrg_gilt_X24_lTRG_gilt_L40.pth --tensor_path data/hotrg_gilt_X24.pth --iLayer 40 --mcf_enabled --gilt_enabled  --svd_num_eigvecs 32
python run_get_ltrg_eigvecs.py --filename data/layers/hotrg_gilt_X24_lTRG_gilt_L42.pth --tensor_path data/hotrg_gilt_X24.pth --iLayer 42 --mcf_enabled --gilt_enabled  --svd_num_eigvecs 32
python run_get_ltrg_eigvecs.py --filename data/layers/hotrg_gilt_X24_lTRG_gilt_L44.pth --tensor_path data/hotrg_gilt_X24.pth --iLayer 44 --mcf_enabled --gilt_enabled  --svd_num_eigvecs 32
python run_get_ltrg_eigvecs.py --filename data/layers/hotrg_gilt_X24_lTRG_gilt_L46.pth --tensor_path data/hotrg_gilt_X24.pth --iLayer 46 --mcf_enabled --gilt_enabled  --svd_num_eigvecs 32
python run_get_ltrg_eigvecs.py --filename data/layers/hotrg_gilt_X24_lTRG_gilt_L48.pth --tensor_path data/hotrg_gilt_X24.pth --iLayer 48 --mcf_enabled --gilt_enabled  --svd_num_eigvecs 32
python run_get_ltrg_eigvecs.py --filename data/layers/hotrg_gilt_X24_lTRG_gilt_L50.pth --tensor_path data/hotrg_gilt_X24.pth --iLayer 50 --mcf_enabled --gilt_enabled  --svd_num_eigvecs 32
python run_get_ltrg_eigvecs.py --filename data/layers/hotrg_gilt_X24_lTRG_gilt_L52.pth --tensor_path data/hotrg_gilt_X24.pth --iLayer 52 --mcf_enabled --gilt_enabled  --svd_num_eigvecs 32
python run_get_ltrg_eigvecs.py --filename data/layers/hotrg_gilt_X24_lTRG_gilt_L54.pth --tensor_path data/hotrg_gilt_X24.pth --iLayer 54 --mcf_enabled --gilt_enabled  --svd_num_eigvecs 32
python run_get_ltrg_eigvecs.py --filename data/layers/hotrg_gilt_X24_lTRG_gilt_L56.pth --tensor_path data/hotrg_gilt_X24.pth --iLayer 56 --mcf_enabled --gilt_enabled  --svd_num_eigvecs 32
python run_get_ltrg_eigvecs.py --filename data/layers/hotrg_gilt_X24_lTRG_gilt_L58.pth --tensor_path data/hotrg_gilt_X24.pth --iLayer 58 --mcf_enabled --gilt_enabled  --svd_num_eigvecs 32


# n-point functions
python run_gen_correlation_points.py

# critical, lowB, highB
python run_calculate_correlation.py --filename data/hotrg_gilt_X24_correlation.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/hotrg_gilt_X24.pth --log2Size 30
python run_calculate_correlation.py --filename data/hotrg_gilt_X24_lowB_correlation.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/hotrg_gilt_X24_lowB.pth --log2Size 30
python run_calculate_correlation.py --filename data/hotrg_gilt_X24_highB_correlation.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/hotrg_gilt_X24_highB.pth --log2Size 30

python run_calculate_observables.py --output_filename data/hotrg_gilt_X24_observables.csv --tensors_filename data/hotrg_gilt_X24.pth --observables magnetization --double_layer --device
python run_calculate_observables.py --output_filename data/hotrg_gilt_X24_lowB_observables.csv --tensors_filename data/hotrg_gilt_X24_lowB.pth --observables magnetization --double_layer
python run_calculate_observables.py --output_filename data/hotrg_gilt_X24_highB_observables.csv --tensors_filename data/hotrg_gilt_X24_highB.pth --observables magnetization --double_layer


# nogilt, 00, nogilt 00
python run_calculate_correlation.py --filename data/hotrg_X24_correlation.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/hotrg_X24.pth --log2Size 30
python run_calculate_correlation.py --filename data/hotrg_gilt_X24_correlation_00.pkl --points_filename data/2pt_correlation_points_30_00.pkl --tensors_filename data/hotrg_gilt_X24.pth --log2Size 30
python run_calculate_correlation.py --filename data/hotrg_X24_correlation_00.pkl --points_filename data/2pt_correlation_points_30_00.pkl --tensors_filename data/hotrg_X24.pth --log2Size 30

# additional points for improved precision
python run_calculate_correlation.py --filename data/hotrg_gilt_X24_correlation_appended.pkl --points_filename data/2pt_correlation_points_30_appended.pkl --tensors_filename data/hotrg_gilt_X24.pth --log2Size 30
python run_calculate_correlation.py --filename data/hotrg_gilt_X24_lowB_correlation_appended.pkl --points_filename data/2pt_correlation_points_30_appended.pkl --tensors_filename data/hotrg_gilt_X24_lowB.pth --log2Size 30
python run_calculate_correlation.py --filename data/hotrg_gilt_X24_highB_correlation_appended.pkl --points_filename data/2pt_correlation_points_30_appended.pkl --tensors_filename data/hotrg_gilt_X24_highB.pth --log2Size 30

# four points
python run_calculate_correlation.py --filename data/hotrg_gilt_X24_4pt_correlation_10.pkl --points_filename data/4pt_correlation_points_10.pkl --tensors_filename data/hotrg_gilt_X24.pth --log2Size 10
python run_calculate_correlation.py --filename data/hotrg_gilt_X24_4pt_correlation.pkl --points_filename data/4pt_correlation_points.pkl --tensors_filename data/hotrg_gilt_X24.pth --log2Size 30

# smearing corners
python run_calculate_correlation.py --filename data/hotrg_gilt_X24_smearing_corner_10.pkl --points_filename data/smearing_corner_10.pkl --tensors_filename data/hotrg_gilt_X24.pth --log2Size 10

# torus, with and without gilt. scan along y
python run_calculate_correlation.py --filename data/hotrg_X24_torus_correlation_y_10.pkl --points_filename data/torus_correlation_points_y_10.pkl --tensors_filename data/hotrg_X24.pth --log2Size 10
python run_calculate_correlation.py --filename data/hotrg_gilt_X24_torus_correlation_y_10.pkl --points_filename data/torus_correlation_points_y_10.pkl --tensors_filename data/hotrg_gilt_X24.pth --log2Size 10

# data/torus_correlation_points_y_mid_10.pkl
python run_calculate_correlation.py --filename data/hotrg_X24_torus_correlation_y_mid_10.pkl --points_filename data/torus_correlation_points_y_mid_10.pkl --tensors_filename data/hotrg_X24.pth --log2Size 10
python run_calculate_correlation.py --filename data/hotrg_gilt_X24_torus_correlation_y_mid_10.pkl --points_filename data/torus_correlation_points_y_mid_10.pkl --tensors_filename data/hotrg_gilt_X24.pth --log2Size 10

# torus big, y=0
# data/torus_correlation_points_30_00.pkl
python run_calculate_correlation.py --filename data/hotrg_X24_torus_correlation_30_00.pkl --points_filename data/torus_correlation_points_30_00.pkl --tensors_filename data/hotrg_X24.pth --log2Size 30
python run_calculate_correlation.py --filename data/hotrg_gilt_X24_torus_correlation_30_00.pkl --points_filename data/torus_correlation_points_30_00.pkl --tensors_filename data/hotrg_gilt_X24.pth --log2Size 30

# torus big, y=mid
# data/torus_correlation_points_30_mid.pkl
python run_calculate_correlation.py --filename data/hotrg_X24_torus_correlation_30_mid.pkl --points_filename data/torus_correlation_points_30_mid.pkl --tensors_filename data/hotrg_X24.pth --log2Size 30
python run_calculate_correlation.py --filename data/hotrg_gilt_X24_torus_correlation_30_mid.pkl --points_filename data/torus_correlation_points_30_mid.pkl --tensors_filename data/hotrg_gilt_X24.pth --log2Size 30









# study what is the max dimension
#  --gilt_enabled --mcf_enabled
# python run_HOTRG.py --filename data/.trash/trash.pth --nLayers 15 --max_dim 46 --model Ising2D --params '{"beta":0.440674}' --overwrite --gilt_enabled --mcf_enabled

# 40**5 * 8byte=784MB 
# 4859 MB HOTRG only 4883 GILT+HOTRG 4861 GILT+HOTRG+MCF

# 46**5*8=1571MB
