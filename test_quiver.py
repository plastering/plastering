from plastering.inferencers.quiver import DummyQuiver
import rdflib



ebu3b_ttl = 'groundtruth/ebu3b_brick.ttl'
ebu3b_g = rdflib.Graph()
ebu3b_g.parse(ebu3b_ttl, format='turtle')


quiver = DummyQuiver('ebu3b', [], config={'ground_truth_ttl':ebu3b_ttl})
#quiver.update_prior(ebu3b_g)
pred = quiver.predict()

