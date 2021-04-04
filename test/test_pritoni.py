from plastering.inferencers.quiver import DummyPritoni
import rdflib


ebu3b_ttl = '../groundtruth/ebu3b_brick.ttl'
ebu3b_g = rdflib.Graph()
ebu3b_g.parse(ebu3b_ttl, format='turtle')


pritoni = DummyPritoni(ebu3b_ttl, 'ebu3b', [])
pritoni.update_prior(ebu3b_g)
pred = pritoni.predict()

