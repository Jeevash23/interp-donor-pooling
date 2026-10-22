"""Analytic examples catch linear, curvature, scoring and ranking failure modes."""
import unittest
import numpy as np
from analyze_core import reversal

class AnalyticTests(unittest.TestCase):
    def setUp(self):
        self.x=np.array([[1.,2.],[-2.,1.],[3.,-4.],[.5,2.]])
        self.mu=self.x.mean(0);self.d=self.x-self.mu
    def test_affine_pooling_commutes(self):
        f=lambda x:x@np.array([2.,-3.])+4.
        self.assertAlmostEqual(f(self.mu),f(self.x).mean(),places=12)
    def test_quadratic_gap_is_covariance_contraction(self):
        h=np.array([[3.,1.],[1.,-2.]])
        f=lambda x:np.einsum('...i,ij,...j->...',x,h,x)/2+x@np.array([2.,1.])
        cov=self.d.T@self.d/len(self.x)
        actual=f(self.mu)-f(self.x).mean()
        self.assertAlmostEqual(actual,-np.trace(h@cov)/2,places=12)
        for t in [.25,.5,.75]:self.assertAlmostEqual(f(self.mu)-f(self.mu+t*self.d).mean(),t*t*actual,places=12)
    def test_cubic_remainder_prevents_universal_quadratic_claim(self):
        x=np.array([-2.,0.,3.]);mu=x.mean();f=lambda z:z**3
        full=f(mu)-f(x).mean();half=f(mu)-f(mu+.5*(x-mu)).mean()
        self.assertGreater(abs(full-4*half),.01)
    def test_softmax_has_gap_without_propagation_or_normalization(self):
        logits=np.array([[4.,0.],[0.,0.]])
        p=lambda z:np.exp(z[...,0]-np.logaddexp(z[...,0],z[...,1]))
        self.assertGreater(abs(p(logits.mean(0))-p(logits).mean()),.01)
        self.assertAlmostEqual((logits.mean(0)[0]-logits.mean(0)[1])-(logits[:,0]-logits[:,1]).mean(),0.)
    def test_ordered_decomposition_closes_and_can_cancel(self):
        a,b,c,e=2.,5.,1.,3.
        terms=np.array([a-b,b-c,c-e])
        self.assertEqual(terms.sum(),a-e)
        self.assertGreater(abs(terms).sum(),abs(a-e))
    def test_rank_margin_rejects_near_ties(self):
        self.assertTrue(reversal(np.array([1.,0.]),np.array([0.,1.]),.05)[0])
        self.assertFalse(reversal(np.array([.001,0.]),np.array([0.,1.]),.05)[0])
        self.assertFalse(reversal(np.array([1.,0.]),np.array([2.,0.]),.05)[0])
    def test_zero_dispersion(self):
        x=np.tile(self.mu,(8,1));f=lambda z:np.exp(z.sum(-1))
        self.assertAlmostEqual(f(x.mean(0)),f(x).mean(),places=12)

if __name__=="__main__":unittest.main()
