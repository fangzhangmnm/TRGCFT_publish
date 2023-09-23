# Ising2D analytical critical point 0.44068679350977147

NLAYERS=60
MAX_DIM=24
# beta_critical=0.44068381958007813

# take beta_critical as 0.440684

BETAS=(
    "0.440684"
    "0.440683" "0.440682" "0.440681" "0.440680" "0.440679" "0.440678" "0.440677" "0.440676" "0.440675"
    "0.440685" "0.440686" "0.440687" "0.440688" "0.440689" "0.440690" "0.440691" "0.440692" "0.440693"
    "0.440674" "0.440664" "0.440654" "0.440644" "0.440634" "0.440624" "0.440614" "0.440604" "0.440594"
    "0.440694" "0.440704" "0.440714" "0.440724" "0.440734" "0.440744" "0.440754" "0.440764" "0.440774"
    "0.440784" "0.440884" "0.440984" "0.441084" "0.441184" "0.441284" "0.441384" "0.441484" "0.441584"
    "0.440574" "0.440474" "0.440374" "0.440274" "0.440174" "0.440074" "0.439974" "0.439874" "0.439774"
)

for BETA in "${BETAS[@]}";
do
    python run_HOTRG.py --filename data/hotrg_gilt_X${MAX_DIM}_betas/${BETA}.pth --nLayers ${NLAYERS} --max_dim ${MAX_DIM} --gilt_enabled --mcf_enabled --model Ising2D --params '{"beta":'${BETA}'}'
    python run_calculate_correlation.py --filename data/hotrg_gilt_X${MAX_DIM}_betas/${BETA}_correlation.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/hotrg_gilt_X${MAX_DIM}_betas/${BETA}.pth --log2Size 30
    python run_calculate_observables.py --output_filename data/hotrg_gilt_X${MAX_DIM}_betas/${BETA}_observables.csv --tensors_filename data/hotrg_gilt_X${MAX_DIM}_betas/${BETA}.pth --observables magnetization --double_layer
done

# 2pt_correlation_points_30_appended

for BETA in "${BETAS[@]}";
do
    python run_calculate_correlation.py --filename data/hotrg_gilt_X${MAX_DIM}_betas/${BETA}_correlation_appended.pkl --points_filename data/2pt_correlation_points_30_appended.pkl --tensors_filename data/hotrg_gilt_X${MAX_DIM}_betas/${BETA}.pth --log2Size 30
done