# @title Setup - Import relevant modules

# The following code imports relevant modules that
# allow you to run the colab.
# If you encounter technical issues running some of the code sections
# that follow, try running this section again.

import pandas as pd

# The following lines adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

#@title Import the dataset

# The following code imports the dataset that is used in the colab.

training_df = pd.read_csv(filepath_or_buffer="https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")

training_df.describe()

# @title Solution (run this code block to view) { display-mode: "form" }

print("""The following columns might contain outliers:

  * total_rooms
  * total_bedrooms
  * population
  * households
  * possibly, median_income

In all of those columns:

  * the standard deviation is almost as high as the mean
  * the delta between 75% and max is much higher than the
      delta between min and 25%.""")