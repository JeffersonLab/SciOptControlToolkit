# Create data circle
import numpy as np
def circle_rdm_samples(ndim=2, nsamples=100, std_min=1, rcut=0.5, give_all=False):
  all_vectors = []
  vectors = []
  ntrials = 0
  while len(vectors)<(nsamples):
    ntrials += 1
    norm_dim = np.random.normal(0,std_min, ndim+2)
    norm = np.sum(norm_dim*norm_dim)**(0.5)
    vector = [norm_dim[i]/norm for i in range(ndim)]
    r = np.sqrt(sum([v*v for v in vector]))
    if r>rcut:
      vectors.append(vector)
      all_vectors.append(norm_dim)
  if give_all==True:
    return np.array(vectors).flatten(), len(vectors)/ntrials, np.array(all_vectors).flatten()
  return np.array(vectors).flatten(), len(vectors)/ntrials

# Create data circle
import numpy as np
def ring_rdm_samples(ndim=2, nsamples=100, rcut_max=1.0, rcut_min=0.5, give_all=False):
  all_vectors = []
  vectors = []
  ntrials = 0
  while len(vectors)<(nsamples):
    ntrials += 1
    norm_dim = np.random.normal(0,1, ndim+2)
    norm = np.sum(norm_dim*norm_dim)**(0.5)
    vector = [norm_dim[i]/norm for i in range(ndim)]
    r = np.sqrt(sum([v*v for v in vector]))
    if r>rcut_min and r<rcut_max:
      vectors.append(vector)
      all_vectors.append(norm_dim)
  if give_all==True:
    return np.array(vectors).flatten(), len(vectors)/ntrials, np.array(all_vectors).flatten()
  return np.array(vectors).flatten(), len(vectors)/ntrials