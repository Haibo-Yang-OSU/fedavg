
import sys
from absl import flags
flags.DEFINE_integer('device', 0, 'CUDA device.')

FLAGS = flags.FLAGS

FLAGS(sys.argv)
flags.DEFINE_list('addnum', [], 'CUDA device.')
FLAGS.device = 2
FLAGS.addnum = list(range(3))
print(FLAGS.device, FLAGS.addnum)

import numpy as np
b = np.load('test.npy', allow_pickle=True)