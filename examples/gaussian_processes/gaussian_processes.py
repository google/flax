from flax import struct
from distributions import GaussianProcess
import distributions
from inducing_variables import InducingVariable


@struct.dataclass
class VariationalGaussianProcess(GaussianProcess):
    inducing_variable: InducingVariable

    def prior_kl(self):
        qu = self.inducing_variable.variational_distribution
        pu = self.inducing_variable.prior_distribution
        return distributions.multivariate_gaussian_kl(qu, pu)