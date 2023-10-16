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
    python run_get_ltrg_eigvecs.py --filename data/layers/hotrg_gilt_X24_lTRG_gilt_L${LAYER}.pth --tensor_path data/hotrg_gilt_X24.pth --iLayer ${LAYER} --mcf_enabled  --gilt_enabled --gilt_eps ${GILT_EPS}  --svd_num_eigvecs 32 --device ${DEVICE}
done

for LAYER in "${LAYERS[@]}"
do
    python run_get_ltrg_eigvecs.py --filename data/layers/hotrg_X24_lTRG_L${LAYER}.pth --tensor_path data/hotrg_X24.pth --iLayer ${LAYER} --mcf_enabled --svd_num_eigvecs 32 --device ${DEVICE}
done