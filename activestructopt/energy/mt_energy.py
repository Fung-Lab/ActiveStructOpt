from activestructopt.common.registry import registry
from activestructopt.energy.base import BaseEnergy

def hparams(radius = 6.0, max_num_neighbors = 120):
  import mattertune.configs as MC
  import mattertune as mt

  hparams = MC.MatterTunerConfig.draft()

  # Model hparams
  hparams.model = MC.ORBBackboneConfig.draft()
  hparams.model.system = mt.backbones.orb.model.ORBSystemConfig(
    radius = radius, max_num_neighbors = max_num_neighbors)
  hparams.model.pretrained_model = "orb-v3-direct-inf-omat"

  hparams.model.ignore_gpu_batch_transform_error = True
  hparams.model.reset_output_heads = False
  hparams.model.reset_backbone = False

  hparams.model.properties = []
  energy = MC.EnergyPropertyConfig(name = 'spectra', 
    dtype = 'float', loss = MC.MAELossConfig(),)
  hparams.model.properties.append(energy)

  hparams = hparams.finalize()
  return hparams

@registry.register_dataset("MTEnergy")
class MTEnergy(BaseEnergy):
  def __init__(self, radius = 6.0, max_num_neighbors = 120):
    from mattertune import MatterTuner
    hp = hparams(radius = radius, max_num_neighbors = max_num_neighbors)
    tune_output = MatterTuner(hp).tune()
    self.model = tune_output.model.to('cuda')

  def get(self, batch):
    return self.model.model_forward(batch)['predicted_properties']['energy']
    