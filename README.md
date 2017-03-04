### Software and Libraries used in the project

Python libraries tensorflow, sqlite3, pandas, itertools, numpy, sklearn, argparse, and matplotlib need to be installed. 

The first script to run is `database.py` that collects the data off the web and stores it into an `articles.sqlite` database.

The second script `count_db.py` counts the number of articles per author and stores that result in a table in the `articles.sqlite` database.

Note that the full database has already been uploded to this repository to be used by `classify.py` script.

The main script `classify.py` first loads the data from the `articles.sqlite` database and runs the benchmark logistic regression model and then a convolutional neural network model, computes the accuracy score using a test dataset. For a CNN model, the script stores the result to be visualized using TensorBoard. 

After the script is run succesfully, you can run the following command to visualize the accuracy and cross entropy measures for training and testing datasets:

```
tensorboard --logdir=/tmp/tensorflow/logs
```

The `capstone.ipynb` contains additional data exploration visualizations including Pareto chart and word cloud, PCA, logistic regression model and a CNN model.