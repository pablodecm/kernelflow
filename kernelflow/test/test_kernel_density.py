
from kernelflow.kernel_density import KernelDensity
from tensorflow.python.ops.distributions import normal
import tensorflow as tf
import numpy as np

def samples_one_normal(sample_shape, loc=0.5, scale=0.1):
    norm = normal.Normal(loc,scale)
    return norm.sample(sample_shape)

def _test_one_kernel(loc, scale, weight=None, kernel_dist=normal.Normal):
    one_kernel = kernel_dist(loc,scale) 
    kde = KernelDensity(loc=loc, scale=scale, weight=weight,
                        kernel_dist = kernel_dist)
    x = tf.constant(one_kernel.sample(10).eval())
    if weight is None:
        assert np.allclose(kde._w_lp.eval(), 0.)
    assert np.allclose(one_kernel.log_prob(x).eval(), kde.log_prob(x).eval())
    assert np.allclose(one_kernel.log_cdf(x).eval(), kde.log_cdf(x).eval())

def _test_several_kernel(loc, scale, weight=None, kernel_dist=normal.Normal):
    one_kernel = kernel_dist(loc[0:1],scale[0:1]) 
    kde = KernelDensity(loc=loc, scale=scale, weight=weight,
                        kernel_dist = kernel_dist)
    n_samples = 10
    x = tf.constant(one_kernel.sample(n_samples).eval())
    if weight is None:
        assert np.allclose(kde._w_lp.eval(), 0.)
    assert kde.log_prob(x).eval().shape == (n_samples, 1) 
    assert kde.log_cdf(x).eval().shape == (n_samples, 1) 
    assert np.greater_equal(kde.prob(x).eval(), 0.0).all() 
    assert np.greater_equal(kde.cdf(x).eval(), 0.0).all()
    assert np.less_equal(kde.cdf(x).eval(), 1.0).all()


class test_kernel_density(tf.test.TestCase):

  def test_1d(self):
    with self.test_session():
        _test_one_kernel(np.array([0.5]),np.array([0.3]))
        _test_several_kernel(np.array([0.2,0.8]),np.array([0.3,0.3]))
        _test_several_kernel(np.linspace(0.1,0.8,20),np.array([0.01]))
        _test_several_kernel(np.linspace(0.1,0.8,20),np.array([0.01]*20))


  def test_1d_weighted(self):
    with self.test_session():
        _test_one_kernel(np.array([0.5]),np.array([0.3]),weight=np.array([0.1]))
        _test_several_kernel(np.array([0.2,0.8]),np.array([0.3,0.3]),
                             weight=np.array([0.1,0.9]))
        n_s = 10 
        _test_several_kernel(np.random.random(n_s),
                             np.array([0.01]), weight=np.random.random(n_s))

