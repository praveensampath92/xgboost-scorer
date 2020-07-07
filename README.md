# xgboost-scorer

**xgboost-scorer** is a pure JavaScript scoring function implementation for [XGBoost](https://github.com/dmlc/xgboost) models, for use in a Node.js environment or even in a browser.

**NOTE:**  This implementation is an extremely naive one, that will most likely not work well with large inputs. Also, it doesn't handle all scenarios (currently only works for binary classification, but can be easily extended for other types of tasks). If performance is a big concern, consider using the [xgboost-node](https://github.com/nuanio/xgboost-node) package instead, which is a [Node.js addon](https://nodejs.org/api/addons.html) that invokes the actual XGBoost scoring function. However, you might have to compile the package for your specific platform (e.g., Windows), which could be *a bit of a problem*. So, if you don't care about serving a million QPS, are working with small models and inputs; and want to run in multiple hostile environments (e.g., in the browser), then this package might be of use to you.

## Installation

    npm install xgboost-scorer

## Usage

    import Scorer from 'xgboost-scorer';

    // Create a scorer using an XGBoost model and (optionally) a feature index
    const scorer = await Scorer.create('xgboost.model.json', 'feature_index.json');

    // Get the score for an instance by passing a mapping of (feature -> value)
    const score =
	    await scorer.score({
            featureA: 100,
            featureB: 0.1,
            categoricalFeatureC: 1.0
	    })
    scorePromise.then(console.log); // E.g., 0.42

## Documentation
A scorer is created by specifying the XGBoost model to use, in JSON. You can either create it using a JSON file on disk (in Node.js-like environments) or by passing in the JSON object directly (in the browser, for instance):

    // In Node.js
    const scorer = await Scorer.create('/path/to/xgboost.model.json');

    // In the browser (or even in Node.js)
    const scorer = await Scorer.create([
	  {
	    "nodeid": 0,
	    "depth": 0,
	    "split": "top_diff",
	    "split_condition": 19.40625,
	    "yes": 1,
	    "no": 2,
	    ...
    ])

This will handle most use-cases that require scoring in an online setting. For offline scoring, you would typically have your input data to be scored in a file in [LibSVM format](https://xgboost.readthedocs.io/en/latest/tutorials/input_format.html), and this is where the feature index comes in:

    const scorer = Scorer.create('/path/to/xgboost.model.json', '/path/to/feature_index.json')
    scorer.score('/path/to/data.libsvm');

The feature index is required to provide a mapping between the feature names (present in your JSON model dump) and the feature indices (used in the LibSVM file).

### The model file format
We use a JSON dump of the XGBoost model for scoring, because that is the probably what you would want to work with in Node.js (and also because I'm too lazy to figure out how to get this to work with other formats). This is how you would get your model in JSON (if you were doing training/testing in Python, for instance):

    import xgboost as xgb
    ...
    model = xgb.train(...)
    model.dump_model('xgb.model.json', 'feature_map.txt', dump_format='json')

    # This is what it would look like:
    [
	  {
        "nodeid": 0,
        "depth": 0,
        "split": "featureA",
        "split_condition": 22,
        "yes": 1,
        "no": 2,
        "missing": 1,
        "children": [
          {
            "nodeid": 1,
            "leaf": 0.0700000003
          },
          {
            "nodeid": 2,
            "leaf": -0.0700000003
          }
        ]
      }
    ]

### The feature index format
(Only used in offline scoring using a data file in LibSVM format)
The feature index file contains a mapping from feature name to the feature ID used in the LibSVM file, e.g.:

    // feature_index.json
    {
      "featureA": 1,
      "featureB": 2,
      "categoricalFeatureC": 3,
      ...
    }




