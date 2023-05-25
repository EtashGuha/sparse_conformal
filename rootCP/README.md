
# rootCP

This package implements a root-finding approach for computing conformal prediction set without data splitting. See our paper https://arxiv.org/pdf/2104.06648.pdf for more details.


# Example computing conformal set for Ridge regression
```python
import numpy as np
from sklearn.datasets import make_regression
from rootcp import rootCP, models

n_samples, n_features = (300, 50)
X, y = make_regression(n_samples=n_samples, n_features=n_features)
X /= np.linalg.norm(X, axis=0)
y = (y - y.mean()) / y.std()
lmd = 0.5

ridge_regressor = models.ridge(lmd=lmd)
cp = rootCP.conformalset(X, y[:-1], ridge_regressor)
print("CP set is", cp)
```

## Installation & Requirements

The compilation proceed as follows:

```
$ pip install -e .
```

This package has the following requirements:

- [numpy](http://numpy.org)
- [scipy](http://scipy.org)
- [scikit-learn](http://scikit-learn.org) (optional but recommended for generating the examples)