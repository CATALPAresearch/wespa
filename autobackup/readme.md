# todo

- implement full backup pipeline
- add validation functions
- implement pipeline with **luigi**: https://luigi.readthedocs.io/en/stable/workflows.html
- implement it for multiple courses > group.js file
- store network files on polaris instead of nise81.com
- performance bottlenecks:
  - download of json files
  - save csv files during processing
- run performance tests
- include analytics database
- reimplemet node.js code in python
- implemente it for multipel moodle instances

---

### Boltzman

- Test scores (0-100) → Gaussian-Binary RBM
- Course categories (A, B, C, D) → Multinomial RBM
- Completion status (yes/no) → Binary RBM
- Time spent (minutes) → Gaussian-Binary RBM

```{python}
from learnergy.models.bernoulli import RBM
from learnergy.models.gaussian import GaussianRBM

# Binary RBM
model = RBM(n_visible=784, n_hidden=128,
            steps=1, learning_rate=0.1)
model.fit(train_data, batch_size=128, epochs=10)

# Gaussian RBM for continuous data
gaussian_model = GaussianRBM(n_visible=784, n_hidden=128)
gaussian_model.fit(continuous_data, batch_size=128, epochs=10)
```
