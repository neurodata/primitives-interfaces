"""
    A script to run to chcek if all primitives do not throw any
    warnings or exceptions.
"""

import jhu_primitives as jhu 
import _pickle as pickle

print("TESTING: SeededGraphMatching")

truth = pickle.load(open('49_truth.pkl', 'rb'))
train = pickle.load(open('49_train.pkl', 'rb'))
test = pickle.load(open('49_test.pkl', 'rb'))

hp_sgm = jhu.sgm.sgm.Hyperparams().defaults()
SGM = jhu.SeededGraphMatching(hyperparams=hp_sgm)

SGM.set_training_data(inputs = train)
SGM.fit()
predictions = SGM.produce(inputs = test).value

truth_labels = np.array(truth['learningData']['match'])[np.array(predictions['d3mIndex']).astype(int)]
preds = np.array(predictions['match']).astype(str) 
np.sum(preds == truth_labels)/len(preds)

print("ACCURACY: " + str(np.sum(preds == truth_labels)/len(preds)))
print("---------")

truth = pickle.load(open('DS_truth.pkl', 'rb'))
train = pickle.load(open('DS_train.pkl', 'rb'))
test = pickle.load(open('DS_test.pkl', 'rb'))


print("TESTING: GMM o ASE pipeline")
hp_lcc = jhu.lcc.lcc.Hyperparams().defaults
hp_ase = jhu.ase.ase.Hyperparams({'use_attributes': True, 'max_dimension': 5, 'which_elbow': 1})
hp_gmm = jhu.gclust.gclust.Hyperparams({'max_clusters': 10})

# Initialize
LCC = jhu.LargestConnectedComponent(hyperparams=hp_lcc)
ASE = jhu.AdjacencySpectralEmbedding(hyperparams=hp_ase)
GMM = jhu.GaussianClustering(hyperparams=hp_gmm)

# Train
lcc_train = LCC.produce(inputs = train).value
ase_train = ASE.produce(inputs = lcc_train).value
GMM.set_training_data(inputs = ase_train)
GMM.fit()

# Test
lcc_test = LCC.produce(inputs = test).value
ase_test = ASE.produce(inputs = lcc_test).value
predictions = GMM.produce(inputs = ase_test).value

truth_labels = np.array(truth['learningData']['classLabel'])[np.array(predictions['d3mIndex']).astype(int) - 1]
preds = np.array(predictions['classLabel'])

print("ACCURACY: " + str(np.sum(preds.astype(str) == truth_labels)/len(preds)))
print("---------") 

print("TESTING: GMM o LSE")
hp_lcc = jhu.lcc.lcc.Hyperparams().defaults
hp_lse = jhu.lse.lse.Hyperparams({'use_attributes': True, 'max_dimension': 5, 'which_elbow': 1})
hp_gmm = jhu.gclust.gclust.Hyperparams({'max_clusters': 10})

# Initialize
LCC = jhu.LargestConnectedComponent(hyperparams=hp_lcc)
LSE = jhu.LaplacianSpectralEmbedding(hyperparams=hp_lse)
GMM = jhu.GaussianClustering(hyperparams=hp_gmm)

# Train
lcc_train = LCC.produce(inputs = train).value
lse_train = LSE.produce(inputs = lcc_train).value
GMM.set_training_data(inputs = lse_train)
GMM.fit()

# Test
lcc_test = LCC.produce(inputs = test).value
lse_test = LSE.produce(inputs = lcc_test).value
predictions = GMM.produce(inputs = lse_test).value

truth_labels = np.array(truth['learningData']['classLabel'])[np.array(predictions['d3mIndex']).astype(int) - 1]
preds = np.array(predictions['classLabel'])

print("ACCURACY: " + str(np.sum(preds.astype(str) == truth_labels)/len(preds)))
print("---------")

print("TESTING: GaussianClustering via SGC primitive")
hp_sgc = jhu.sgc.sgc.Hyperparams().defaults()

SGC = jhu.SpectralGraphClustering(hyperparams=hp_sgc)

SGC.set_training_data(inputs = train)
SGC.fit()
predictions = SGC.produce(inputs = test).value

truth_labels = np.array(truth['learningData']['classLabel'])[np.array(predictions['d3mIndex']).astype(int) - 1]
preds = np.array(predictions['classLabel'])

print("ACCURACY: " + str(np.sum(preds.astype(str) == truth_labels)/len(preds)))
print("---------")

truth = pickle.load(open('LL1_truth.pkl', 'rb'))
train = pickle.load(open('LL1_train.pkl', 'rb'))
test = pickle.load(open('LL1_test.pkl', 'rb'))

print("TESTING: QDA o ASE")
hp_lcc = jhu.lcc.lcc.Hyperparams().defaults
hp_ase = jhu.ase.ase.Hyperparams({'use_attributes': True, 'max_dimension': 5, 'which_elbow': 1})
hp_gclass = jhu.gclass.gclass.Hyperparams().defaults

# Initialize
LCC = jhu.LargestConnectedComponent(hyperparams=hp_lcc)
ASE = jhu.AdjacencySpectralEmbedding(hyperparams=hp_ase)
GCLASS = jhu.GaussianClassification(hyperparams=hp_gclass)

# Train
lcc_train = LCC.produce(inputs = train).value
ase_train = ASE.produce(inputs = lcc_train).value
GCLASS.set_training_data(inputs = ase_train)
GCLASS.fit()

# Test
lcc_test = LCC.produce(inputs = test).value
ase_test = ASE.produce(inputs = lcc_test).value
predictions = GCLASS.produce(inputs = ase_test).value

truth_labels = np.array(truth['learningData']['classLabel'])[np.array(predictions['d3mIndex'])]
preds = np.array(predictions['classLabel'])

print("ACCURACY: " + str(np.sum(preds.astype(str) == truth_labels)/len(preds)))
print("---------")

print("TESTING: QDA o LSE")
hp_lcc = jhu.lcc.lcc.Hyperparams().defaults
hp_lse = jhu.lse.lse.Hyperparams({'use_attributes': True, 'max_dimension': 5, 'which_elbow': 1})
hp_gclass = jhu.gclass.gclass.Hyperparams().defaults

# Initialize
LCC = jhu.LargestConnectedComponent(hyperparams=hp_lcc)
LSE = jhu.LaplacianSpectralEmbedding(hyperparams=hp_lse)
GCLASS = jhu.GaussianClassification(hyperparams=hp_gclass)

# Train
lcc_train = LCC.produce(inputs = train).value
lse_train = LSE.produce(inputs = lcc_train).value
GCLASS.set_training_data(inputs = lse_train)
GCLASS.fit()

# Test
lcc_test = LCC.produce(inputs = test).value
lse_test = LSE.produce(inputs = lcc_test).value
predictions = GCLASS.produce(inputs = lse_test).value

truth_labels = np.array(truth['learningData']['classLabel'])[np.array(predictions['d3mIndex'])]
preds = np.array(predictions['classLabel'])

print("ACCURACY: " + str(np.sum(preds.astype(str) == truth_labels)/len(preds)))
print("---------")

print("TESTING: QDA via SGC")
hp_sgc = jhu.sgc.sgc.Hyperparams().defaults()

SGC = jhu.SpectralGraphClustering(hyperparams=hp_sgc)

SGC.set_training_data(inputs = train)
SGC.fit()
predictions = SGC.produce(inputs = test).value

truth_labels = np.array(truth['learningData']['classLabel'])[np.array(predictions['d3mIndex']).astype(int)]
preds = np.array(predictions['classLabel'])

print("ACCURACY: " + str(np.sum(preds.astype(str) == truth_labels)/len(preds)))
print("---------")

truth = pickle.load(open('EDGELIST_truth.pkl', 'rb'))
train = pickle.load(open('EDGELIST_train.pkl', 'rb'))
test = pickle.load(open('EDGELIST_test.pkl', 'rb'))

print("TESTING: EDGELIST QDA o ASE")
hp_lcc = jhu.lcc.lcc.Hyperparams().defaults
hp_ase = jhu.ase.ase.Hyperparams({'use_attributes': True, 'max_dimension': 5, 'which_elbow': 1})
hp_gclass = jhu.gclass.gclass.Hyperparams().defaults

# Initialize
LCC = jhu.LargestConnectedComponent(hyperparams=hp_lcc)
ASE = jhu.AdjacencySpectralEmbedding(hyperparams=hp_ase)
GCLASS = jhu.GaussianClassification(hyperparams=hp_gclass)

# Train
lcc_train = LCC.produce(inputs = train).value
ase_train = ASE.produce(inputs = lcc_train).value
GCLASS.set_training_data(inputs = ase_train)
GCLASS.fit()

# Test
lcc_test = LCC.produce(inputs = test).value
ase_test = ASE.produce(inputs = lcc_test).value
predictions = GCLASS.produce(inputs = ase_test).value

truth_labels = np.array(truth['learningData']['classLabel'])[np.array(predictions['d3mIndex'])]
preds = np.array(predictions['classLabel'])

print("ACCURACY: " + str(np.sum(preds.astype(str) == truth_labels)/len(preds)))
print("---------")

print("TESTING: EDGELIST QDA o LSE")
hp_lcc = jhu.lcc.lcc.Hyperparams().defaults
hp_lse = jhu.lse.lse.Hyperparams({'use_attributes': True, 'max_dimension': 5, 'which_elbow': 1})
hp_gclass = jhu.gclass.gclass.Hyperparams().defaults

# Initialize
LCC = jhu.LargestConnectedComponent(hyperparams=hp_lcc)
LSE = jhu.LaplacianSpectralEmbedding(hyperparams=hp_lse)
GCLASS = jhu.GaussianClassification(hyperparams=hp_gclass)

# Train
lcc_train = LCC.produce(inputs = train).value
lse_train = LSE.produce(inputs = lcc_train).value
GCLASS.set_training_data(inputs = lse_train)
GCLASS.fit()

# Test
lcc_test = LCC.produce(inputs = test).value
lse_test = LSE.produce(inputs = lcc_test).value
predictions = GCLASS.produce(inputs = lse_test).value

truth_labels = np.array(truth['learningData']['classLabel'])[np.array(predictions['d3mIndex'])]
preds = np.array(predictions['classLabel'])

print("ACCURACY: " + str(np.sum(preds.astype(str) == truth_labels)/len(preds)))
print("---------")

print("TESTING: EDGELIST QDA via SGC")
hp_sgc = jhu.sgc.sgc.Hyperparams().defaults()

SGC = jhu.SpectralGraphClustering(hyperparams=hp_sgc)

SGC.set_training_data(inputs = train)
SGC.fit()
predictions = SGC.produce(inputs = test).value

truth_labels = np.array(truth['learningData']['classLabel'])[np.array(predictions['d3mIndex']).astype(int)]
preds = np.array(predictions['classLabel'])
print("ACCURACY: " + str(np.sum(preds.astype(str) == truth_labels)/len(preds)))
print("---------")

print("TESTING: LinkPrediction")
hp_lpgr = jhu.link_pred_graph_reader.link_pred_graph_reader.Hyperparams().defaults()
hp_ase = jhu.ase.ase.Hyperparams({'which_elbow': 1, 'max_dimension': 2, 'use_attributes': False})
hp_lprc = jhu.link_pred_rc.link_pred_rc.Hyperparams().defaults()

GR = jhu.LinkPredictionGraphReader(hyperparams=hp_lpgr)
ASE = jhu.AdjacencySpectralEmbedding(hyperparams=hp_ase)
RC = jhu.LinkPredictionRankClassifier(hyperparams = hp_lprc)

gr_train = GR.produce(inputs = train).value
ase_train = ASE.produce(inputs = gr_train).value
RC.set_training_data(inputs = ase_train)
RC.fit()

gr_test = GR.produce(inputs = test).value
ase_test = ASE.produce(inputs = gr_test).value
predictions = RC.produce(inputs = ase_test).value

truth_labels = np.array(truth['learningData']['linkExists'])[np.array(predictions['d3mIndex']).astype(int)]
preds = np.array(predictions['linkExists'])
print("ACCURACY: " + str(np.sum(preds.astype(str) == truth_labels)/len(preds)))
print("---------")

# print("TESTING: ")
# print("ACCURACY: " + str(np.sum(preds.astype(str) == truth_labels)/len(preds)))
# print("---------")

# print("TESTING: ")
# print("ACCURACY: " + str(np.sum(preds.astype(str) == truth_labels)/len(preds)))
# print("---------")

# print("TESTING: ")
# print("ACCURACY: " + str(np.sum(preds.astype(str) == truth_labels)/len(preds)))
# print("---------")