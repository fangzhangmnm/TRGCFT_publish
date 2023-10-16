NLAYERS=60
MAX_DIM=24
DEVICE="cuda:0"
GILT_EPS=8e-7

python run_calculate_correlation.py --filename data/hotrg_X24_correlation.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/hotrg_X24.pth --log2Size 30 --device ${DEVICE} --overwrite

python run_calculate_correlation.py --filename data/hotrg_X24_correlation_00.pkl --points_filename data/2pt_correlation_points_30_00.pkl --tensors_filename data/hotrg_X24.pth --log2Size 30 --device ${DEVICE} --overwrite

python run_calculate_correlation.py --filename data/hotrg_X24_torus_correlation_y_10.pkl --points_filename data/torus_correlation_points_y_10.pkl --tensors_filename data/hotrg_X24.pth --log2Size 10 --device ${DEVICE}  --overwrite

python run_calculate_correlation.py --filename data/hotrg_X24_torus_correlation_y_mid_10.pkl --points_filename data/torus_correlation_points_y_mid_10.pkl --tensors_filename data/hotrg_X24.pth --log2Size 10 --device ${DEVICE}--overwrite

python run_calculate_correlation.py --filename data/hotrg_X24_torus_correlation_30_00.pkl --points_filename data/torus_correlation_points_30_00.pkl --tensors_filename data/hotrg_X24.pth --log2Size 30 --device ${DEVICE} --overwrite

python run_calculate_correlation.py --filename data/hotrg_X24_torus_correlation_30_mid.pkl --points_filename data/torus_correlation_points_30_mid.pkl --tensors_filename data/hotrg_X24.pth --log2Size 30 --device ${DEVICE} --overwrite

python run_calculate_correlation.py --filename data/hotrg_X24_smearing_corner_10.pkl --points_filename data/smearing_corner_10.pkl --tensors_filename data/hotrg_X24.pth --log2Size 10 --device ${DEVICE} --overwrite

LAYERS=(
    2 4 6 8 10
    12 14 16 18 20
    22 24 26 28 30
    32 34 36 38 40
    42 44 46 48 50
    52 54 56 58
)

NLAYERS=60
MAX_DIM=24
DEVICE="cuda:0"
GILT_EPS=8e-7


for LAYER in "${LAYERS[@]}"
do
    python run_get_ltrg_eigvecs.py --filename data/layers/hotrg_X24_lTRG_L${LAYER}.pth --tensor_path data/hotrg_X24.pth --iLayer ${LAYER} --mcf_enabled --svd_num_eigvecs 32 --device ${DEVICE} --overwrite
done