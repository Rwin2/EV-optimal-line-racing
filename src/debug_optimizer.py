from src import track
from src.optimizer import SpeedProfileOptimizer
import traceback

try:
    trk = track.get_track('complex')
    opt = SpeedProfileOptimizer(trk.centerline, a_max=4.0, a_brake=8.0)
    v = opt.optimize(w_time=1.0, w_energy=1e-4)
    print('SOLVED', v.shape, v.min(), v.max())
except Exception as e:
    print(type(e).__name__, e)
    traceback.print_exc()
