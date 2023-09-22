# Ising2D analytical critical point 0.44068679350977147

NLAYERS=60
MAX_DIM=24
# beta_critical=0.44068381958007813

# take beta_critical as 0.440684

# scan with interval 0.000010

python run_HOTRG.py --filename data/hotrg_gilt_X${MAX_DIM}_betas/0.440684.pth --nLayers ${NLAYERS} --max_dim ${MAX_DIM} --gilt_enabled --mcf_enabled --model Ising2D --params '{"beta":0.440684}'

python run_HOTRG.py --filename data/hotrg_gilt_X${MAX_DIM}_betas/0.440674.pth --nLayers ${NLAYERS} --max_dim ${MAX_DIM} --gilt_enabled --mcf_enabled --model Ising2D --params '{"beta":0.440674}'
python run_HOTRG.py --filename data/hotrg_gilt_X${MAX_DIM}_betas/0.440664.pth --nLayers ${NLAYERS} --max_dim ${MAX_DIM} --gilt_enabled --mcf_enabled --model Ising2D --params '{"beta":0.440664}'
python run_HOTRG.py --filename data/hotrg_gilt_X${MAX_DIM}_betas/0.440654.pth --nLayers ${NLAYERS} --max_dim ${MAX_DIM} --gilt_enabled --mcf_enabled --model Ising2D --params '{"beta":0.440654}'
python run_HOTRG.py --filename data/hotrg_gilt_X${MAX_DIM}_betas/0.440644.pth --nLayers ${NLAYERS} --max_dim ${MAX_DIM} --gilt_enabled --mcf_enabled --model Ising2D --params '{"beta":0.440644}'
python run_HOTRG.py --filename data/hotrg_gilt_X${MAX_DIM}_betas/0.440634.pth --nLayers ${NLAYERS} --max_dim ${MAX_DIM} --gilt_enabled --mcf_enabled --model Ising2D --params '{"beta":0.440634}'

python run_HOTRG.py --filename data/hotrg_gilt_X${MAX_DIM}_betas/0.440694.pth --nLayers ${NLAYERS} --max_dim ${MAX_DIM} --gilt_enabled --mcf_enabled --model Ising2D --params '{"beta":0.440694}'
python run_HOTRG.py --filename data/hotrg_gilt_X${MAX_DIM}_betas/0.440704.pth --nLayers ${NLAYERS} --max_dim ${MAX_DIM} --gilt_enabled --mcf_enabled --model Ising2D --params '{"beta":0.440704}'
python run_HOTRG.py --filename data/hotrg_gilt_X${MAX_DIM}_betas/0.440714.pth --nLayers ${NLAYERS} --max_dim ${MAX_DIM} --gilt_enabled --mcf_enabled --model Ising2D --params '{"beta":0.440714}'
python run_HOTRG.py --filename data/hotrg_gilt_X${MAX_DIM}_betas/0.440724.pth --nLayers ${NLAYERS} --max_dim ${MAX_DIM} --gilt_enabled --mcf_enabled --model Ising2D --params '{"beta":0.440724}'
python run_HOTRG.py --filename data/hotrg_gilt_X${MAX_DIM}_betas/0.440734.pth --nLayers ${NLAYERS} --max_dim ${MAX_DIM} --gilt_enabled --mcf_enabled --model Ising2D --params '{"beta":0.440734}'


python run_calculate_correlation.py --filename data/hotrg_gilt_X${MAX_DIM}_betas/0.440684_correlation.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/hotrg_gilt_X${MAX_DIM}_betas/0.440684.pth --log2Size 30

python run_calculate_correlation.py --filename data/hotrg_gilt_X${MAX_DIM}_betas/0.440674_correlation.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/hotrg_gilt_X${MAX_DIM}_betas/0.440674.pth --log2Size 30
python run_calculate_correlation.py --filename data/hotrg_gilt_X${MAX_DIM}_betas/0.440664_correlation.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/hotrg_gilt_X${MAX_DIM}_betas/0.440664.pth --log2Size 30
python run_calculate_correlation.py --filename data/hotrg_gilt_X${MAX_DIM}_betas/0.440654_correlation.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/hotrg_gilt_X${MAX_DIM}_betas/0.440654.pth --log2Size 30
python run_calculate_correlation.py --filename data/hotrg_gilt_X${MAX_DIM}_betas/0.440644_correlation.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/hotrg_gilt_X${MAX_DIM}_betas/0.440644.pth --log2Size 30
python run_calculate_correlation.py --filename data/hotrg_gilt_X${MAX_DIM}_betas/0.440634_correlation.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/hotrg_gilt_X${MAX_DIM}_betas/0.440634.pth --log2Size 30

python run_calculate_correlation.py --filename data/hotrg_gilt_X${MAX_DIM}_betas/0.440694_correlation.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/hotrg_gilt_X${MAX_DIM}_betas/0.440694.pth --log2Size 30
python run_calculate_correlation.py --filename data/hotrg_gilt_X${MAX_DIM}_betas/0.440704_correlation.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/hotrg_gilt_X${MAX_DIM}_betas/0.440704.pth --log2Size 30
python run_calculate_correlation.py --filename data/hotrg_gilt_X${MAX_DIM}_betas/0.440714_correlation.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/hotrg_gilt_X${MAX_DIM}_betas/0.440714.pth --log2Size 30
python run_calculate_correlation.py --filename data/hotrg_gilt_X${MAX_DIM}_betas/0.440724_correlation.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/hotrg_gilt_X${MAX_DIM}_betas/0.440724.pth --log2Size 30
python run_calculate_correlation.py --filename data/hotrg_gilt_X${MAX_DIM}_betas/0.440734_correlation.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename data/hotrg_gilt_X${MAX_DIM}_betas/0.440734.pth --log2Size 30


python run_calculate_observables.py --output_filename data/hotrg_gilt_X${MAX_DIM}_betas/0.440684_observables.csv --tensors_filename data/hotrg_gilt_X${MAX_DIM}_betas/0.440684.pth --observables magnetization --double_layer

python run_calculate_observables.py --output_filename data/hotrg_gilt_X${MAX_DIM}_betas/0.440674_observables.csv --tensors_filename data/hotrg_gilt_X${MAX_DIM}_betas/0.440674.pth --observables magnetization --double_layer
python run_calculate_observables.py --output_filename data/hotrg_gilt_X${MAX_DIM}_betas/0.440664_observables.csv --tensors_filename data/hotrg_gilt_X${MAX_DIM}_betas/0.440664.pth --observables magnetization --double_layer
python run_calculate_observables.py --output_filename data/hotrg_gilt_X${MAX_DIM}_betas/0.440654_observables.csv --tensors_filename data/hotrg_gilt_X${MAX_DIM}_betas/0.440654.pth --observables magnetization --double_layer
python run_calculate_observables.py --output_filename data/hotrg_gilt_X${MAX_DIM}_betas/0.440644_observables.csv --tensors_filename data/hotrg_gilt_X${MAX_DIM}_betas/0.440644.pth --observables magnetization --double_layer
python run_calculate_observables.py --output_filename data/hotrg_gilt_X${MAX_DIM}_betas/0.440634_observables.csv --tensors_filename data/hotrg_gilt_X${MAX_DIM}_betas/0.440634.pth --observables magnetization --double_layer

python run_calculate_observables.py --output_filename data/hotrg_gilt_X${MAX_DIM}_betas/0.440694_observables.csv --tensors_filename data/hotrg_gilt_X${MAX_DIM}_betas/0.440694.pth --observables magnetization --double_layer
python run_calculate_observables.py --output_filename data/hotrg_gilt_X${MAX_DIM}_betas/0.440704_observables.csv --tensors_filename data/hotrg_gilt_X${MAX_DIM}_betas/0.440704.pth --observables magnetization --double_layer
python run_calculate_observables.py --output_filename data/hotrg_gilt_X${MAX_DIM}_betas/0.440714_observables.csv --tensors_filename data/hotrg_gilt_X${MAX_DIM}_betas/0.440714.pth --observables magnetization --double_layer
python run_calculate_observables.py --output_filename data/hotrg_gilt_X${MAX_DIM}_betas/0.440724_observables.csv --tensors_filename data/hotrg_gilt_X${MAX_DIM}_betas/0.440724.pth --observables magnetization --double_layer
python run_calculate_observables.py --output_filename data/hotrg_gilt_X${MAX_DIM}_betas/0.440734_observables.csv --tensors_filename data/hotrg_gilt_X${MAX_DIM}_betas/0.440734.pth --observables magnetization --double_layer


