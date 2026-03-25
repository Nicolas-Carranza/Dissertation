# Numerical and Categorical Data, and Generalization and Overfitting

# Numerical data: How a model ingests data using feature vectors

Until now, we've given you the impression that a model acts directly on the rows of a dataset; however, models actually ingest data somewhat differently.

For example, suppose a dataset provides five columns, but only two of those columns (**`b`** and **`d`**) are features in the model. When processing the example in row 3, does the model simply grab the contents of the highlighted two cells (3b and 3d) as follows?

![Figure 1. A model ingesting an example directly from a dataset.
            Columns b and d of Row 3 are highlighted.](https://developers.google.com/static/machine-learning/crash-course/images/dataset_directly_to_model.svg)

***Figure 1.** Not exactly how a model gets its examples.*

In fact, the model actually ingests an array of floating-point values called a [**feature vector**](https://developers.google.com/machine-learning/glossary#feature-vector). You can think of a feature vector as the floating-point values comprising one example.

![Figure 2. The feature vector is an intermediary between the dataset
            and the model.](https://developers.google.com/static/machine-learning/crash-course/images/dataset_to_feature_vector_to_model.svg)

***Figure 2.** Closer to the truth, but not realistic.*

However, feature vectors seldom use the dataset's *raw values*. Instead, you must typically process the dataset's values into representations that your model can better learn from. So, a more realistic feature vector might look something like this:

![Figure 3. The feature vector contains two floating-point values:
            0.13 and 0.47. A more realistic feature vector.](https://developers.google.com/static/machine-learning/crash-course/images/dataset_to_feature_vector_to_model_scaled.svg)

***Figure 3.** A more realistic feature vector.*

Wouldn't a model produce better predictions by training from the *actual* values in the dataset than from *altered* values? Surprisingly, the answer is no.

You must determine the best way to represent raw dataset values as trainable values in the feature vector. This process is called [**feature engineering**](https://developers.google.com/machine-learning/glossary#feature-engineering), and it is a vital part of machine learning. The most common feature engineering techniques are:

- [**Normalization**](https://developers.google.com/machine-learning/glossary#normalization): Converting numerical values into a standard range.
- [**Binning**](https://developers.google.com/machine-learning/glossary#binning) (also referred to as [**bucketing**](https://developers.google.com/machine-learning/glossary#bucketing)): Converting numerical values into buckets of ranges.

This unit covers normalizing and binning. The next unit, [Working with categorical data](https://developers.google.com/machine-learning/crash-course/categorical-data), covers other forms of [**preprocessing**](https://developers.google.com/machine-learning/glossary#preprocessing), such as converting non-numerical data, like strings, to floating point values.

Every value in a feature vector must be a floating-point value. However, many features are naturally strings or other non-numerical values. Consequently, a large part of feature engineering is representing non-numerical values as numerical values. You'll see a lot of this in later modules.

## **Numerical data: First steps**

Before creating feature vectors, we recommend studying numerical data in two ways:

- Visualize your data in plots or graphs.
- Get statistics about your data.

## Visualize your data

Graphs can help you find anomalies or patterns hiding in the data. Therefore, before getting too far into analysis, look at your data graphically, either as scatter plots or histograms. View graphs not only at the beginning of the data pipeline, but also throughout data transformations. Visualizations help you continually check your assumptions.

We recommend working with pandas for visualization:

- [Working with Missing Data (pandas Documentation)](http://pandas.pydata.org/pandas-docs/stable/missing_data.html)
- [Visualizations (pandas Documentation)](http://pandas.pydata.org/pandas-docs/stable/visualization.html)

Note that certain visualization tools are optimized for certain data formats. A visualization tool that helps you evaluate protocol buffers may or may not be able to help you evaluate CSV data.

## Statistically evaluate your data

Beyond visual analysis, we also recommend evaluating potential features and labels mathematically, gathering basic statistics such as:

- mean and median
- standard deviation
- the values at the quartile divisions: the 0th, 25th, 50th, 75th, and 100th percentiles. The 0th percentile is the minimum value of this column; the 100th percentile is the maximum value of this column. (The 50th percentile is the median.)

## Find outliers

An [**outlier**](https://developers.google.com/machine-learning/glossary#outliers) is a value *distant* from most other values in a feature or label. Outliers often cause problems in model training, so finding outliers is important.

When the delta between the 0th and 25th percentiles differs significantly from the delta between the 75th and 100th percentiles, the dataset probably contains outliers.

**Note:** Don't over-rely on basic statistics. Anomalies can also hide in seemingly well-balanced data.

Outliers can fall into any of the following categories:

- The outlier is due to a *mistake*. For example, perhaps an experimenter mistakenly entered an extra zero, or perhaps an instrument that gathered data malfunctioned. You'll generally delete examples containing mistake outliers.
- The outlier is a legitimate data point, *not a mistake*. In this case, will your trained model ultimately need to infer good predictions on these outliers?
    - If yes, keep these outliers in your training set. After all, outliers in certain features sometimes mirror outliers in the label, so the outliers could actually *help* your model make better predictions. Be careful, extreme outliers can still hurt your model.
    - If no, delete the outliers or apply more invasive feature engineering techniques, such as [**clipping**](https://developers.google.com/machine-learning/glossary#clipping).

## **Numerical data: Normalization**

After examining your data through statistical and visualization techniques, you should transform your data in ways that will help your model train more effectively. The goal of [**normalization**](https://developers.google.com/machine-learning/glossary#normalization) is to transform features to be on a similar scale. For example, consider the following two features:

- Feature **`X`** spans the range 154 to 24,917,482.
- Feature **`Y`** spans the range 5 to 22.

These two features span very different ranges. Normalization might manipulate **`X`** and **`Y`** so that they span a similar range, perhaps 0 to 1.

Normalization provides the following benefits:

- Helps models *converge more quickly* during training. When different features have different ranges, gradient descent can "bounce" and slow convergence. That said, more advanced optimizers like [Adagrad](https://developers.google.com/machine-learning/glossary#adagrad) and [Adam](https://arxiv.org/abs/1412.6980) protect against this problem by changing the effective learning rate over time.
- Helps models *infer better predictions*. When different features have different ranges, the resulting model might make somewhat less useful predictions.
- Helps *avoid the "NaN trap"* when feature values are very high. [NaN](https://wikipedia.org/wiki/NaN) is an abbreviation for *not a number*. When a value in a model exceeds the floating-point precision limit, the system sets the value to **`NaN`** instead of a number. When one number in the model becomes a NaN, other numbers in the model also eventually become a NaN.
- Helps the model *learn appropriate weights* for each feature. Without feature scaling, the model pays too much attention to features with wide ranges and not enough attention to features with narrow ranges.

We recommend normalizing numeric features covering distinctly different ranges (for example, age and income). We also recommend normalizing a single numeric feature that covers a wide range, such as **`city population.`**

Warning: If you normalize a feature during training, you must also normalize that feature when making predictions.

Consider the following two features:

- Feature **`A`**'s lowest value is -0.5 and highest is +0.5.
- Feature **`B`**'s lowest value is -5.0 and highest is +5.0.

Feature **`A`** and Feature **`B`** have relatively narrow spans. However, Feature **`B`**'s span is 10 times wider than Feature **`A`**'s span. Therefore:

- At the start of training, the model assumes that Feature **`B`** is ten times more "important" than Feature **`A`**.
- Training will take longer than it should.
- The resulting model may be suboptimal.

The overall damage due to not normalizing will be relatively small; however, we still recommend normalizing Feature A and Feature B to the same scale, perhaps -1.0 to +1.0.

Now consider two features with a greater disparity of ranges:

- Feature **`C`**'s lowest value is -1 and highest is +1.
- Feature **`D`**'s lowest value is +5000 and highest is +1,000,000,000.

If you don't normalize Feature **`C`** and Feature **`D`**, your model will likely be suboptimal. Furthermore, training will take much longer to converge or even fail to converge entirely!

This section covers three popular normalization methods:

- linear scaling
- Z-score scaling
- log scaling

This section additionally covers [**clipping**](https://developers.google.com/machine-learning/glossary#clipping). Although not a true normalization technique, clipping does tame unruly numerical features into ranges that produce better models.

## Linear scaling

[**Linear scaling**](https://developers.google.com/machine-learning/glossary#scaling) (more commonly shortened to just **scaling**) means converting floating-point values from their natural range into a standard range—usually 0 to 1 or -1 to +1.

Use the following formula to scale to the standard range 0 to 1, inclusive:

$$
x^′=(x−x_{min})/(x_{max}−x_{min})
$$

where:

- $x^′$  is the scaled value.
- $x$ is the original value.
- $x_{min}$ is the lowest value in the dataset of this feature.
- $x_{max}$ is the highest value in the dataset of this feature.

For example, consider a feature named **`quantity`** whose natural range spans 100 to 900. Suppose the natural value of **`quantity`** in a particular example is 300. Therefore, you can calculate the normalized value of 300 as follows:

- $x = 300$
- $x_{min} = 100$
- $x_{max} = 900$

$$
x' = \frac {(300 - 100)} {(900 - 100)} \\
x' = \frac {200}{800} \\
x' = 0.25

$$

Linear scaling is a good choice when all of the following conditions are met:

- The lower and upper bounds of your data don't change much over time.
- The feature contains few or no outliers, and those outliers aren't extreme.
- The feature is approximately uniformly distributed across its range. That is, a histogram would show roughly even bars for most values.

Suppose human **`age`** is a feature. Linear scaling is a good normalization technique for **`age`** because:

- The approximate lower and upper bounds are 0 to 100.
- **`age`** contains a relatively small percentage of outliers. Only about 0.3% of the population is over 100.
- Although certain ages are somewhat better represented than others, a large dataset should contain sufficient examples of all ages.

## Z-score scaling

A **Z-score** is the number of standard deviations a value is from the mean. For example, a value that is 2 standard deviations *greater* than the mean has a Z-score of +2.0. A value that is 1.5 standard deviations *less* than the mean has a Z-score of -1.5.

Representing a feature with **Z-score scaling** means storing that feature's Z-score in the feature vector. For example, the following figure shows two histograms:

- On the left, a classic normal distribution.
- On the right, the same distribution normalized by Z-score scaling.

![Figure 4. Two histograms: both showing normal distributions with
           the identical distribution. The first histogram, which contains raw
           data, has a mean of 200 and a standard deviation of 30. The second
           histogram, which contains a Z-score version of the first
           distribution, has a mean of 0 and a standard deviation of 1.](https://developers.google.com/static/machine-learning/crash-course/images/z-scaling_classic.png)

***Figure 4.** Raw data (left) versus Z-score (right) for a normal distribution.*

Z-score scaling is also a good choice for data like that shown in the following figure, which has only a vaguely normal distribution.

![Figure 5. Two histograms of identical shape, each showing a steep
            rise to a plateau and then a relatively quick descent followed by
            gradual decay. One histogram illustrates the
            distribution of the raw data; the other histogram illustrates the
            distribution of the raw data when normalized by Z-score scaling.
            The values on the X-axis of the two histograms are very different.
            The raw data histogram spans the domain 0 to 29,000, while
            the Z-score scaled histogram ranges from -1 to about +4.8](https://developers.google.com/static/machine-learning/crash-course/images/z-scaling-non-classic-normal-distribution.png)

***Figure 5.** Raw data (left) versus Z-score scaling (right) for a non-classic normal distribution.*

Use the following formula to normalize a value, x, to its Z-score:

$$
x^′=\frac{x-μ}{σ}
$$

where:

- $x^′$  is the Z-score.
- $x$ is the raw value; that is, $x$, is the value you are normalizing.
- $μ$ is the mean.
- $σ$ is the standard deviation.

For example, suppose:

- $mean = 100$
- $standard \ deviation = 20$
- $original \ value = 130$

Therefore:

$$
  Z \ score = \frac {(130 - 100)}{20} \\
  Z \ score = \frac {30}{20} \\
  Z \ score = +1.5
$$

In a classic normal distribution:

- At least 68.27% of data has a Z-score between -1.0 and +1.0.
- At least 95.45% of data has a Z-score between -2.0 and +2.0.
- At least 99.73% of data has a Z-score between -3.0 and +3.0.
- At least 99.994% of data has a Z-score between -4.0 and +4.0.

So, data points with a Z-score less than -4.0 or more than +4.0 are rare, but are they truly outliers? Since *outliers* is a concept without a strict definition, no one can say for sure. Note that a dataset with a sufficiently large number of examples will almost certainly contain at least a few of these "rare" examples. For example, a feature with one billion examples conforming to a classic normal distribution could have as many as 60,000 examples with a score outside the range -4.0 to +4.0.

Z-score is a good choice when the data follows a normal distribution or a distribution somewhat like a normal distribution.

Note that some distributions might be normal within the bulk of their range, but still contain extreme outliers. For example, nearly all of the points in a **`net_worth`** feature might fit neatly into 3 standard deviations, but a few examples of this feature could be hundreds of standard deviations away from the mean. In these situations, you can combine Z-score scaling with another form of normalization (usually clipping) to handle this situation.

## Log scaling

Log scaling computes the logarithm of the raw value. In theory, the logarithm could be any base; in practice, log scaling usually calculates the natural logarithm (ln).

Use the following formula to normalize a value, x, to its log:

$$
x^′=ln(x)
$$

where:

- $x^′$ is the natural logarithm of $x$
- $original\ value = 54.598$

Therefore, the log of the original value is about 4.0:

$$
  4.0 = ln(54.598)
$$

Log scaling is helpful when the data conforms to a *power law* distribution. Casually speaking, a power law distribution looks as follows:

- Low values of **`X`** have very high values of **`Y`**.
- As the values of **`X`** increase, the values of **`Y`** quickly decrease. Consequently, high values of **`X`** have very low values of **`Y`**.

Movie ratings are a good example of a power law distribution. In the following figure, notice:

- A few movies have lots of user ratings. (Low values of **`X`** have high values of **`Y`**.)
- Most movies have very few user ratings. (High values of **`X`** have low values of **`Y`**.)

Log scaling changes the distribution, which helps train a model that will make better predictions.

![Figure 6. Two graphs comparing raw data versus the log of raw data.
            The raw data graph shows a lot of user ratings in the head, followed
            by a long tail. The log graph has a more even distribution.](https://developers.google.com/static/machine-learning/data-prep/images/norm-log-scaling-movie-ratings.svg)

***Figure 6.** Comparing a raw distribution to its log.*

As a second example, book sales conform to a power law distribution because:

- Most published books sell a tiny number of copies, maybe one or two hundred.
- Some books sell a moderate number of copies, in the thousands.
- Only a few *bestsellers* will sell more than a million copies.

Suppose you are training a linear model to find the relationship of, say, book covers to book sales. A linear model training on raw values would have to find something about book covers on books that sell a million copies that is 10,000 more powerful than book covers that sell only 100 copies. However, log scaling all the sales figures makes the task far more feasible. For example, the log of 100 is:

- $~4.6 = ln(100)$

while the log of 1,000,000 is:

- $~13.8 = ln(1,000,000)$

So, the log of 1,000,000 is only about three times larger than the log of 100. You probably *could* imagine a bestseller book cover being about three times more powerful (in some way) than a tiny-selling book cover.

## Clipping

[**Clipping**](https://developers.google.com/machine-learning/glossary#clipping) is a technique to minimize the influence of extreme outliers. In brief, clipping usually caps (reduces) the value of outliers to a specific maximum value. Clipping is a strange idea, and yet, it can be very effective.

For example, imagine a dataset containing a feature named **`roomsPerPerson`**, which represents the number of rooms (total rooms divided by number of occupants) for various houses. The following plot shows that over 99% of the feature values conform to a normal distribution (roughly, a mean of 1.8 and a standard deviation of 0.7). However, the feature contains a few outliers, some of them extreme:

![Figure 7. A plot of roomsPerPerson in which nearly all the values
            are clustered between 0 and 4, but there's a verrrrry long tail
            reaching all the way out to 17 rooms per person](https://developers.google.com/static/machine-learning/crash-course/images/PreClipping.png)

***Figure 7.** Mainly normal, but not completely normal.*

How can you minimize the influence of those extreme outliers? Well, the histogram is not an even distribution, a normal distribution, or a power law distribution. What if you simply *cap* or *clip* the maximum value of **`roomsPerPerson`** at an arbitrary value, say 4.0?

![A plot of roomsPerPerson in which all values lie between 0 and
            4.0. The plot is bell-shaped, but there's an anomalous hill at 4.0](https://developers.google.com/static/machine-learning/crash-course/images/Clipping.png)

***Figure 8.** Clipping feature values at 4.0.*

Clipping the feature value at 4.0 doesn't mean that your model ignores all values greater than 4.0. Rather, it means that all values that were greater than 4.0 now become 4.0. This explains the peculiar hill at 4.0. Despite that hill, the scaled feature set is now more useful than the original data.

Wait a second! Can you really reduce every outlier value to some arbitrary upper threshold? When training a model, yes.

You can also clip values after applying other forms of normalization. For example, suppose you use Z-score scaling, but a few outliers have absolute values far greater than 3. In this case, you could:

- Clip Z-scores greater than 3 to become exactly 3.
- Clip Z-scores less than -3 to become exactly -3.

Clipping prevents your model from overindexing on unimportant data. However, some outliers are actually important, so clip values carefully.

# Summary of normalization techniques

The best normalization technique is one that works well in practice, so try new ideas if you think they'll work well on your feature distribution.

| **Normalization technique** | **Formula** | **When to use** |
| --- | --- | --- |
| Linear scaling | $x'=\frac {x−x_{min}}{x_{max}−x_{min}}$ | When the feature is mostly uniformly distributed across range. **Flat-shaped** |
| Z-score scaling | $x′=\frac{x−μ}{σ}$ | When the feature is normally distributed (peak close to mean). **Bell-shaped** |
| Log scaling | $x′=log(x)$ | When the feature distribution is heavy skewed on at least either side of tail. **Heavy Tail-shaped** |
| Clipping | $If \ x > max, set x′=max \\ If x<min, set x′=min$ | When the feature contains extreme outliers. |

## **Numerical data: Binning**

**Binning** (also called **bucketing**) is a [**feature engineering**](https://developers.google.com/machine-learning/glossary#feature_engineering) technique that groups different numerical subranges into *bins* or [***buckets***](https://developers.google.com/machine-learning/glossary#bucketing). In many cases, binning turns numerical data into categorical data. For example, consider a [**feature**](https://developers.google.com/machine-learning/glossary#feature) named **`X`** whose lowest value is 15 and highest value is 425. Using binning, you could represent **`X`** with the following five bins:

- Bin 1: 15 to 34
- Bin 2: 35 to 117
- Bin 3: 118 to 279
- Bin 4: 280 to 392
- Bin 5: 393 to 425

Bin 1 spans the range 15 to 34, so every value of **`X`** between 15 and 34 ends up in Bin 1. A model trained on these bins will react no differently to **`X`** values of 17 and 29 since both values are in Bin 1.

The [**feature vector**](https://developers.google.com/machine-learning/glossary#feature_vector) represents the five bins as follows:

| **Bin number** | **Range** | **Feature vector** |
| --- | --- | --- |
| 1 | 15-34 | [1.0, 0.0, 0.0, 0.0, 0.0] |
| 2 | 35-117 | [0.0, 1.0, 0.0, 0.0, 0.0] |
| 3 | 118-279 | [0.0, 0.0, 1.0, 0.0, 0.0] |
| 4 | 280-392 | [0.0, 0.0, 0.0, 1.0, 0.0] |
| 5 | 393-425 | [0.0, 0.0, 0.0, 0.0, 1.0] |

Even though **`X`** is a single column in the dataset, binning causes a model to treat **`X`** as *five* separate features. Therefore, the model learns separate weights for each bin.

Binning is a good alternative to [**scaling**](https://developers.google.com/machine-learning/glossary#scaling) or [**clipping**](https://developers.google.com/machine-learning/glossary#clipping) when either of the following conditions is met:

- The overall *linear* relationship between the feature and the [**label**](https://developers.google.com/machine-learning/glossary#label) is weak or nonexistent.
- When the feature values are clustered.

Binning can feel counterintuitive, given that the model in the previous example treats the values 37 and 115 identically. But when a feature appears more *clumpy* than linear, binning is a much better way to represent the data.

# Binning example: number of shoppers versus temperature

Suppose you are creating a model that predicts the number of shoppers by the outside temperature for that day. Here's a plot of the temperature versus the number of shoppers:

![Figure 9. A scatter plot of 45 points. The 45 points naturally
            cluster into three groups.](https://developers.google.com/static/machine-learning/crash-course/images/binning_temperature_vs_shoppers.png)

***Figure 9.** A scatter plot of 45 points.*

The plot shows, not surprisingly, that the number of shoppers was highest when the temperature was most comfortable.

You could represent the feature as raw values: a temperature of 35.0 in the dataset would be 35.0 in the feature vector. Is that the best idea?

During training, a linear regression model learns a single weight for each feature. Therefore, if temperature is represented as a single feature, then a temperature of 35.0 would have five times the influence (or one-fifth the influence) in a prediction as a temperature of 7.0. However, the plot doesn't really show any sort of linear relationship between the label and the feature value.

The graph suggests three clusters in the following subranges:

- Bin 1 is the temperature range 4-11.
- Bin 2 is the temperature range 12-26.
- Bin 3 is the temperature range 27-36.

![Figure 10. The same scatter plot of 45 points as in the previous
            figure, but with vertical lines to make the bins more obvious.](https://developers.google.com/static/machine-learning/crash-course/images/binning_temperature_vs_shoppers_divided_into_3_bins.png)

***Figure 10.** The scatter plot divided into three bins.*

The model learns separate weights for each bin.

While it's possible to create more than three bins, even a separate bin for each temperature reading, this is often a bad idea for the following reasons:

- A model can only learn the association between a bin and a label if there are enough examples in that bin. In the given example, each of the 3 bins contains at least 10 examples, which *might* be sufficient for training. With 33 separate bins, none of the bins would contain enough examples for the model to train on.
- A separate bin for each temperature results in 33 separate temperature features. However, you typically should *minimize* the number of features in a model.

# Quantile Bucketing

**Quantile bucketing** creates bucketing boundaries such that the number of examples in each bucket is exactly or nearly equal. Quantile bucketing mostly hides the outliers.

To illustrate the problem that quantile bucketing solves, consider the equally spaced buckets shown in the following figure, where each of the ten buckets represents a span of exactly 10,000 dollars. Notice that the bucket from 0 to 10,000 contains dozens of examples but the bucket from 50,000 to 60,000 contains only 5 examples. Consequently, the model has enough examples to train on the 0 to 10,000 bucket but not enough examples to train on for the 50,000 to 60,000 bucket.

![Figure 13. A plot of car price versus the number of cars sold at
            that price. The number of cars sold peaks at a price of 6,000.
            Above a price of 6,000, the number of cars sold generally
            decreases, with very few cars sold between a price of 40,000 to
            60,000. The plot is divided into 6 equally-sized buckets, each with
            a range of 10,000. So, the first bucket contains all the cars sold
            between a price of 0 and a price of 10,000, the second
            bucket contains all the cars sold between a price of 10,001 and
            20,000, and so on. The first bucket contain many examples; each
            subsequent bucket contains fewer examples.](https://developers.google.com/static/machine-learning/crash-course/images/NeedsQuantileBucketing.png)

***Figure 13.** Some buckets contain a lot of cars; other buckets contain very few cars.*

In contrast, the following figure uses quantile bucketing to divide car prices into bins with approximately the same number of examples in each bucket. Notice that some of the bins encompass a narrow price span while others encompass a very wide price span.

![Figure 14. Same as previous figure, except with quantile buckets.
            That is, the buckets now have different sizes. The first bucket
            contains the cars sold from 0 to 4,000, the second bucket contains
            the cars sold from 4,001 to 6,000. The sixth bucket contains the
            cars sold from 25,001 to 60,000. The number of cars in each bucket
            is now about the same.](https://developers.google.com/static/machine-learning/crash-course/images/QuantileBucketing.png)

***Figure 14.** Quantile bucketing gives each bucket about the same number of cars.*

Bucketing with equal intervals works for many data distributions. For skewed data, however, try quantile bucketing. Equal intervals give extra information space to the long tail while compacting the large torso into a single bucket. Quantile buckets give extra information space to the large torso while compacting the long tail into a single bucket.

# Numerical data: Scrubbing

Apple trees produce a mixture of great fruit and wormy messes. Yet the apples in high-end grocery stores display 100% perfect fruit. Between orchard and grocery, someone spends significant time removing the bad apples or spraying a little wax on the salvageable ones. As an ML engineer, you'll spend enormous amounts of your time tossing out bad examples and cleaning up the salvageable ones. Even a few bad apples can spoil a large dataset.

Many examples in datasets are unreliable due to one or more of the following problems:

| **Problem category** | **Example** |
| --- | --- |
| Omitted values | A census taker fails to record a resident's age. |
| Duplicate examples | A server uploads the same logs twice. |
| Out-of-range feature values. | A human accidentally types an extra digit. |
| Bad labels | A human evaluator mislabels a picture of an oak tree as a maple. |

You can write a program or script to detect any of the following problems:

- Omitted values
- Duplicate examples
- Out-of-range feature values

For example, the following dataset contains six repeated values:

![Figure 15. The first six values are repeated. The final eight
            values are not.](https://developers.google.com/static/machine-learning/crash-course/images/ScrubDuplicateValues.png)

***Figure 15.** The first six values are repeated.*

As another example, suppose the temperature range for a certain feature must be between 10 and 30 degrees, inclusive. But accidents happen—perhaps a thermometer is temporarily exposed to the sun which causes a bad outlier. Your program or script must identify temperature values less than 10 or greater than 30:

![Figure 16. Nineteen in-range values and one out-of-range value.](https://developers.google.com/static/machine-learning/crash-course/images/ScrubOutofRangeValues.png)

***Figure 16.** An out-of-range value.*

When labels are generated by multiple people, we recommend statistically determining whether each rater generated equivalent sets of labels. Perhaps one rater was a harsher grader than the other raters or used a different set of grading criteria?

Once detected, you typically "fix" examples that contain bad features or bad labels by removing them from the dataset or imputing their values. For details, see the [Data characteristics](https://developers.google.com/machine-learning/crash-course/overfitting/data-characteristics) section of the [Datasets, generalization, and overfitting](https://developers.google.com/machine-learning/crash-course/overfitting) module.

# Numerical data: Qualities of good numerical features

This unit has explored ways to map raw data into suitable [**feature vectors**](https://developers.google.com/machine-learning/glossary#feature_vector). Good numerical [**features**](https://developers.google.com/machine-learning/glossary#feature) share the qualities described in this section.

# Clearly named

Each feature should have a clear, sensible, and obvious meaning to any human on the project. For example, the meaning of the following feature value is confusing:

**Not recommended**

> house_age: 851472000
> 

In contrast, the following feature name and value are far clearer:

**Recommended**

> house_age_years: 27
> 

**Note:** Although your co-workers will rebel against confusing feature and label names, the model won't care (assuming you normalize values properly).

# Checked or tested before training

Although this module has devoted a lot of time to [**outliers**](https://developers.google.com/machine-learning/glossary#outliers), the topic is important enough to warrant one final mention. In some cases, bad data (rather than bad engineering choices) causes unclear values. For example, the following **`user_age_in_years`** came from a source that didn't check for appropriate values:

**Not recommended**

> user_age_in_years: 224
> 

But people *can* be 24 years old:

**Recommended**

> user_age_in_years: 24
> 

Check your data!

# Sensible

A "magic value" is a purposeful discontinuity in an otherwise continuous feature. For example, suppose a continuous feature named **`watch_time_in_seconds`** can hold any floating-point value between 0 and 30 but represents the *absence* of a measurement with the magic value -1:

**Not recommended**

> watch_time_in_seconds: -1
> 

A **`watch_time_in_seconds`** of -1 would force the model to try to figure out what it means to watch a movie backwards in time. The resulting model would probably not make good predictions.

A better technique is to create a separate Boolean feature that indicates whether or not a **`watch_time_in_seconds`** value is supplied. For example:

**Recommended**

> watch_time_in_seconds: 4.82
> 
> 
> is_watch_time_in_seconds_defined=True
> 
> watch_time_in_seconds: 0
> 
> is_watch_time_in_seconds_defined=False
> 

This is a way to handle a continuous dataset with missing values. Now consider a [**discrete**](https://developers.google.com/machine-learning/glossary#discrete_feature) numerical feature, like **`product_category`**, whose values must belong to a finite set of values. In this case, when a value is missing, signify that missing value using a new value in the finite set. With a discrete feature, the model will learn different weights for each value, including original weights for missing features.

For example, we can imagine possible values fitting in the set:

> `{0: 'electronics', 1: 'books', 2: 'clothing', 3: 'missing_category'}.`
> 

# Numerical data: Polynomial transform

Sometimes, when the ML practitioner has domain knowledge suggesting that one variable is related to the square, cube, or other power of another variable, it's useful to create a [**synthetic feature**](https://developers.google.com/machine-learning/glossary#synthetic_feature) from one of the existing numerical [**features**](https://developers.google.com/machine-learning/glossary#feature).

Consider the following spread of data points, where pink circles represent one class or category (for example, a species of tree) and green triangles another class (or species of tree):

![Figure 17. y=x^2 spread of data points, with triangles below the
            curve and circles above the curve.](https://developers.google.com/static/machine-learning/crash-course/images/ft_cross2.png)

***Figure 17.** Two classes that can't be separated by a line.*

It's not possible to draw a straight line that cleanly separates the two classes, but it *is* possible to draw a curve that does so:

![Figure 18. Same image as Figure 17, only this time with y=x^2
            overlaid to create a clear boundary between the triangles and
            circles.](https://developers.google.com/static/machine-learning/crash-course/images/ft_cross1.png)

***Figure 18.** Separating the classes with y = x2.*

As discussed in the [Linear regression module](https://developers.google.com/machine-learning/crash-course/linear-regression), a linear model with one feature, x1, is described by the linear equation:

y=b+w1x1

Additional features are handled by the addition of terms w2x2, w3x3, etc.

[**Gradient descent**](https://developers.google.com/machine-learning/glossary#gradient_descent) finds the [**weight**](https://developers.google.com/machine-learning/glossary#weight) w1 (or weights w1, w2, w3, in the case of additional features) that minimizes the loss of the model. But the data points shown cannot be separated by a line. What can be done?

It's possible to keep both the linear equation *and* allow nonlinearity by defining a new term, x2, that is simply x1 squared:

x2=x12

This synthetic feature, called a polynomial transform, is treated like any other feature. The previous linear formula becomes:

y=b+w1x1+w2x2

This can still be treated like a [**linear regression**](https://developers.google.com/machine-learning/glossary#linear_regression) problem, and the weights determined through gradient descent, as usual, despite containing a hidden squared term, the polynomial transform. Without changing how the linear model trains, the addition of a polynomial transform allows the model to separate the data points using a curve of the form y=b+w1x+w2x2.

Usually the numerical feature of interest is multiplied by itself, that is, raised to some power. Sometimes an ML practitioner can make an informed guess about the appropriate exponent. For example, many relationships in the physical world are related to squared terms, including acceleration due to gravity, the attenuation of light or sound over distance, and elastic potential energy.

If you transform a feature in a way that changes its scale, you should consider experimenting with normalizing it as well. Normalizing after transforming might make the model perform better. For more information, see [Numerical Data: Normalization](https://developers.google.com/machine-learning/crash-course/numerical-data/normalization).

A related concept in [**categorical data**](https://developers.google.com/machine-learning/glossary#categorical_data) is the [**feature cross**](https://developers.google.com/machine-learning/glossary#feature_cross), which more frequently synthesizes two different features.

A machine learning (ML) model's health is determined by its data. Feed your model healthy data and it will thrive; feed your model junk and its predictions will be worthless.

Best practices for working with numerical data:

- Remember that your ML model interacts with the data in the [**feature vector**](https://developers.google.com/machine-learning/glossary#feature_vector), not the data in the [**dataset**](https://developers.google.com/machine-learning/glossary#dataset).
- [**Normalize**](https://developers.google.com/machine-learning/glossary#normalization) most numerical [**features**](https://developers.google.com/machine-learning/glossary#feature).
- If your first normalization strategy doesn't succeed, consider a different way to normalize your data.
- [**Binning**](https://developers.google.com/machine-learning/glossary#binning), also referred to as [**bucketing**](https://developers.google.com/machine-learning/glossary#bucketing), is sometimes better than normalizing.
- Considering what your data *should* look like, write verification tests to validate those expectations. For example:
    - The absolute value of latitude should never exceed 90. You can write a test to check if a latitude value greater than 90 appears in your data.
    - If your data is restricted to the state of Florida, you can write tests to check that the latitudes fall between 24 through 31, inclusive.
- Visualize your data with scatter plots and histograms. Look for anomalies.
- Gather statistics not only on the entire dataset but also on smaller subsets of the dataset. That's because aggregate statistics sometimes obscure problems in smaller sections of a dataset.
- Document all your data transformations.

Data is your most valuable resource, so treat it with care.

## Additional Information

- The *Rules of Machine Learning* guide contains a valuable [Feature Engineering](https://developers.google.com/machine-learning/rules-of-ml/#ml_phase_ii_feature_engineering) section.

# **Working with categorical data**

[**Categorical data**](https://developers.google.com/machine-learning/glossary#categorical-data) has a *specific set* of possible values. For example:

- The different species of animals in a national park
- The names of streets in a particular city
- Whether or not an email is spam
- The colors that house exteriors are painted
- Binned numbers, which are described in the [Working with Numerical Data](https://developers.google.com/machine-learning/crash-course/numerical-data) module

## Numbers can also be categorical data

True [**numerical data**](https://developers.google.com/machine-learning/glossary#numerical-data) can be meaningfully multiplied. For example, consider a model that predicts the value of a house based on its area. Note that a useful model for evaluating house prices typically relies on hundreds of features. That said, all else being equal, a house of 200 square meters should be roughly twice as valuable as an identical house of 100 square meters.

Oftentimes, you should represent features that contain integer values as categorical data instead of numerical data. For example, consider a postal code feature in which the values are integers. If you represent this feature numerically rather than categorically, you're asking the model to find a numeric relationship between different postal codes. That is, you're telling the model to treat postal code 20004 as twice (or half) as large a signal as postal code 10002. Representing postal codes as categorical data lets the model weight each individual postal code separately.

## Encoding

**Encoding** means converting categorical or other data to numerical vectors that a model can train on. This conversion is necessary because models can only train on floating-point values; models can't train on strings such as **`"dog"`** or **`"maple"`**. This module explains different encoding methods for categorical data.

## **Categorical data: Vocabulary and one-hot encoding**

The term **dimension** is a synonym for the number of elements in a [**feature vector**](https://developers.google.com/machine-learning/glossary#feature_vector). Some categorical features are low dimensional. For example:

| **Feature name** | **# of categories** | **Sample categories** |
| --- | --- | --- |
| snowed_today | 2 | True, False |
| skill_level | 3 | Beginner, Practitioner, Expert |
| season | 4 | Winter, Spring, Summer, Autumn |
| day_of_week | 7 | Monday, Tuesday, Wednesday |
| planet | 8 | Mercury, Venus, Earth |

When a categorical feature has a low number of possible categories, you can encode it as a **vocabulary**. With a vocabulary encoding, the model treats each possible categorical value as a *separate feature*. During training, the model learns different weights for each category.

For example, suppose you are creating a model to predict a car's price based, in part, on a categorical feature named **`car_color`**. Perhaps red cars are worth more than green cars. Since manufacturers offer a limited number of exterior colors, **`car_color`** is a low-dimensional categorical feature. The following illustration suggests a vocabulary (possible values) for **`car_color`**:

![Figure 1. Each color in the palette is represented as a separate
      feature. That is, each color is a separate feature in the feature vector.
      For instance, 'Red' is a feature, 'Orange' is a separate feature,
      and so on.](https://developers.google.com/static/machine-learning/crash-course/images/categorical-netview.svg)

***Figure 1.** A unique feature for each category.*

## Index numbers

Machine learning models can only manipulate floating-point numbers. Therefore, you must convert each string to a unique index number, as in the following illustration:

![Figure 2. Each color is associated with a unique integer value. For
      example, 'Red' is associated with the integer 0, 'Orange' with the
      integer 1, and so on.](https://developers.google.com/static/machine-learning/crash-course/images/categorical-netview-indexed.svg)

***Figure 2.** Indexed features.*

After converting strings to unique index numbers, you'll need to process the data further to represent it in ways that help the model learn meaningful relationships between the values. If the categorical feature data is left as indexed integers and loaded into a model, the model would treat the indexed values as continuous floating-point numbers. The model would then consider "purple" six times more likely than "orange."

## One-hot encoding

The next step in building a vocabulary is to convert each index number to its [**one-hot encoding**](https://developers.google.com/machine-learning/glossary#one-hot_encoding). In a one-hot encoding:

- Each category is represented by a vector (array) of N elements, where N is the number of categories. For example, if **`car_color`** has eight possible categories, then the one-hot vector representing will have eight elements.
- Exactly *one* of the elements in a one-hot vector has the value 1.0; all the remaining elements have the value 0.0.

For example, the following table shows the one-hot encoding for each color in **`car_color`**:

| **Feature** | **Red** | **Orange** | **Blue** | **Yellow** | **Green** | **Black** | **Purple** | **Brown** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| "Red" | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| "Orange" | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| "Blue" | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| "Yellow" | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| "Green" | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| "Black" | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| "Purple" | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| "Brown" | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |

It is the one-hot vector, not the string or the index number, that gets passed to the feature vector. The model learns a separate weight for each element of the feature vector.

**Note:** In a true one-hot encoding, only one element has the value 1.0. In a variant known as **multi-hot encoding**, multiple values can be 1.0.

The following illustration suggests the various transformations in the vocabulary representation:

![Figure 3. Diagram of the end-to-end process to map categories to
      feature vectors. In the diagram, the input features are 'Yellow',
      'Orange', 'Blue', and 'Blue' a second time.  The system uses a stored
      vocabulary ('Red' is 0, 'Orange' is 1, 'Blue' is 2, 'Yellow' is 3, and
      so on) to map the input value to an ID. Thus, the system maps 'Yellow',
      'Orange', 'Blue', and 'Blue' to 3, 1, 2, 2. The system then converts
      those values to a one-hot feature vector. For example, given a system
      with eight possible colors, 3 becomes 0, 0, 0, 1, 0, 0, 0, 0.](https://developers.google.com/static/machine-learning/crash-course/images/vocabulary-index-sparse-feature.svg)

***Figure 3.** The end-to-end process to map categories to feature vectors.*

### Sparse representation

A feature whose values are predominantly zero (or empty) is termed a [**sparse feature**](https://developers.google.com/machine-learning/glossary#sparse-feature). Many categorical features, such as **`car_color`**, tend to be sparse features. [**Sparse representation**](https://developers.google.com/machine-learning/glossary#sparse-representation) means storing the *position* of the 1.0 in a sparse vector. For example, the one-hot vector for **`"Blue"`** is:

> [0, 0, 1, 0, 0, 0, 0, 0]
> 

Since the **`1`** is in position 2 (when starting the count at 0), the sparse representation for the preceding one-hot vector is:

> 2
> 

Notice that the sparse representation consumes far less memory than the eight-element one-hot vector. Importantly, the model must *train* on the one-hot vector, not the sparse representation.

**Note:** The sparse representation of a multi-hot encoding stores the positions of *all* the nonzero elements. For example, the sparse representation of a car that is both **`"Blue"`** and **`"Black"`** is **`2, 5`**.

## Outliers in categorical data

Like numerical data, categorical data also contains outliers. Suppose **`car_color`** contains not only the popular colors, but also some rarely used outlier colors, such as **`"Mauve"`** or **`"Avocado"`**. Rather than giving each of these outlier colors a separate category, you can lump them into a single "catch-all" category called *out-of-vocabulary (OOV)*. In other words, all the outlier colors are binned into a single outlier bucket. The system learns a single weight for that outlier bucket.

## Encoding high-dimensional categorical features

Some categorical features have a high number of dimensions, such as those in the following table:

| **Feature name** | **# of categories** | **Sample categories** |
| --- | --- | --- |
| words_in_english | ~500,000 | "happy", "walking" |
| US_postal_codes | ~42,000 | "02114", "90301" |
| last_names_in_Germany | ~850,000 | "Schmidt", "Schneider" |

When the number of categories is high, one-hot encoding is usually a bad choice. *Embeddings*, detailed in a separate [Embeddings module](https://developers.google.com/machine-learning/crash-course/embeddings), are usually a much better choice. Embeddings substantially reduce the number of dimensions, which benefits models in two important ways:

- The model typically trains faster.
- The built model typically infers predictions more quickly. That is, the model has lower latency.

[**Hashing**](https://developers.google.com/machine-learning/glossary#hashing) (also called the *hashing trick*) is a less common way to reduce the number of dimensions.

In brief, hashing maps a category (for example, a color) to a small integer—the number of the "bucket" that will hold that category.

In detail, you implement a hashing algorithm as follows:

1. Set the number of bins in the vector of categories to N, where N is less than the total number of remaining categories. As an arbitrary example, say N = 100.
2. Choose a hash function. (Often, you will choose the range of hash values as well.)
3. Pass each category (for example, a particular color) through that hash function, generating a hash value, say 89237.
4. Assign each bin an index number of the output hash value modulo N. In this case, where N is 100 and the hash value is 89237, the modulo result is 37 because 89237 % 100 is 37.
5. Create a one-hot encoding for each bin with these new index numbers.

For more details about hashing data, see the [Randomization](https://developers.google.com/machine-learning/crash-course/production-ml-systems/monitoring#randomization) section of the [Production machine learning systems](https://developers.google.com/machine-learning/crash-course/production-ml-systems/monitoring) module.

## **Categorical data: Common issues**

Numerical data is often recorded by scientific instruments or automated measurements. Categorical data, on the other hand, is often categorized by human beings or by machine learning (ML) models. *Who* decides on categories and labels, and *how* they make those decisions, affects the reliability and usefulness of that data.

## Human raters

Data manually labeled by human beings is often referred to as *gold labels*, and is considered more desirable than machine-labeled data for training models, due to relatively better data quality.

This doesn't necessarily mean that any set of human-labeled data is of high quality. Human errors, bias, and malice can be introduced at the point of data collection or during data cleaning and processing. Check for them before training.

Any two human beings may label the same example differently. The difference between human raters' decisions is called [**inter-rater agreement**](https://developers.google.com/machine-learning/glossary#inter-rater-agreement). You can get a sense of the variance in raters' opinions by using multiple raters per example and measuring inter-rater agreement.

The following are ways to measure inter-rater agreement:

- Cohen's kappa and variants
- Intra-class correlation (ICC)
- Krippendorff's alpha

For details on Cohen's kappa and intra-class correlation, see [Hallgren 2012](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3402032/). For details on Krippendorff's alpha, see [Krippendorff 2011](https://www.asc.upenn.edu/sites/default/files/2021-03/Computing%20Krippendorff%27s%20Alpha-Reliability.pdf).

## Machine raters

Machine-labeled data, where categories are automatically determined by one or more classification models, is often referred to as *silver labels*. Machine-labeled data can vary widely in quality. Check it not only for accuracy and biases but also for violations of common sense, reality, and intention. For example, if a computer-vision model mislabels a photo of a [chihuahua as a muffin](https://www.freecodecamp.org/news/chihuahua-or-muffin-my-search-for-the-best-computer-vision-api-cbda4d6b425d/), or a photo of a muffin as a chihuahua, models trained on that labeled data will be of lower quality.

Similarly, a sentiment analyzer that scores neutral words as -0.25, when 0.0 is the neutral value, might be scoring all words with an additional negative bias that is not actually present in the data. An oversensitive toxicity detector may falsely flag many neutral statements as toxic. Try to get a sense of the quality and biases of machine labels and annotations in your data before training on it.

## High dimensionality

Categorical data tends to produce high-dimensional feature vectors; that is, feature vectors having a large number of elements. High dimensionality increases training costs and makes training more difficult. For these reasons, ML experts often seek ways to reduce the number of dimensions prior to training.

For natural-language data, the main method of reducing dimensionality is to convert feature vectors to embedding vectors. This is discussed in the [Embeddings module](https://developers.google.com/machine-learning/crash-course/embeddings) later in this course.

## **Categorical data: Feature crosses**

[**Feature crosses**](https://developers.google.com/machine-learning/glossary#feature-cross) are created by crossing (taking the Cartesian product of) two or more categorical or bucketed features of the dataset. Like [polynomial transforms](https://developers.google.com/machine-learning/crash-course/numerical-data/polynomial-transforms), feature crosses allow linear models to handle nonlinearities. Feature crosses also encode interactions between features.

For example, consider a leaf dataset with the categorical features:

- **`edges`**, containing values **`smooth`**, **`toothed`**, and **`lobed`**
- **`arrangement`**, containing values **`opposite`** and **`alternate`**

Assume the order above is the order of the feature columns in a one-hot representation, so that a leaf with **`smooth`** edges and **`opposite`** arrangement is represented as **`{(1, 0, 0), (1, 0)}`**.

The feature cross, or Cartesian product, of these two features would be:

**`{Smooth_Opposite, Smooth_Alternate, Toothed_Opposite, Toothed_Alternate, Lobed_Opposite, Lobed_Alternate}`**

where the value of each term is the product of the base feature values, such that:

- **`Smooth_Opposite = edges[0] * arrangement[0]`**
- **`Smooth_Alternate = edges[0] * arrangement[1]`**
- **`Toothed_Opposite = edges[1] * arrangement[0]`**
- **`Toothed_Alternate = edges[1] * arrangement[1]`**
- **`Lobed_Opposite = edges[2] * arrangement[0]`**
- **`Lobed_Alternate = edges[2] * arrangement[1]`**

For example, if a leaf has a **`lobed`** edge and an **`alternate`** arrangement, the feature-cross vector will have a value of 1 for **`Lobed_Alternate`**, and a value of 0 for all other terms:

**`{0, 0, 0, 0, 0, 1}`**

This dataset could be used to classify leaves by tree species, since these characteristics do not vary within a species.

Feature crosses are somewhat analogous to [Polynomial transforms](https://developers.google.com/machine-learning/crash-course/numerical-data/polynomial-transforms). Both combine multiple features into a new synthetic feature that the model can train on to learn nonlinearities. Polynomial transforms typically combine numerical data, while feature crosses combine categorical data.

## When to use feature crosses

Domain knowledge can suggest a useful combination of features to cross. Without that domain knowledge, it can be difficult to determine effective feature crosses or polynomial transforms by hand. It's often possible, if computationally expensive, to use [neural networks](https://developers.google.com/machine-learning/crash-course/neural-networks) to *automatically* find and apply useful feature combinations during training.

Be careful, crossing two sparse features produces an even sparser new feature than the two original features. For example, if feature A is a 100-element sparse feature and feature B is a 200-element sparse feature, a feature cross of A and B yields a 20,000-element sparse feature.

# **Datasets**

## **Data characteristics**

A [**dataset**](https://developers.google.com/machine-learning/glossary#dataset) is a collection of [**examples**](https://developers.google.com/machine-learning/glossary#example).

Many datasets store data in tables (grids), for example, as comma-separated values (CSV) or directly from spreadsheets or database tables. Tables are an intuitive input format for machine learning [**models**](https://developers.google.com/machine-learning/glossary#model). You can imagine each row of the table as an example and each column as a potential feature or label. That said, datasets may also be derived from other formats, including log files and protocol buffers.

Regardless of the format, your ML model is only as good as the data it trains on. This section examines key data characteristics.

### Types of data

A dataset could contain many kinds of datatypes, including but certainly not limited to:

- numerical data, which is covered in a [separate unit](https://developers.google.com/machine-learning/crash-course/numerical-data)
- categorical data, which is covered in a [separate unit](https://developers.google.com/machine-learning/crash-course/categorical-data)
- human language, including individual words and sentences, all the way up to entire text documents
- multimedia (such as images, videos, and audio files)
- outputs from other ML systems
- [**embedding vectors**](https://developers.google.com/machine-learning/glossary#embedding-vector), which are covered in a later unit

### Quantity of data

As a rough rule of thumb, your model should train on at least an order of magnitude (or two) more examples than trainable parameters. However, good models generally train on *substantially* more examples than that.

Models trained on large datasets with few [**features**](https://developers.google.com/machine-learning/glossary#feature) generally outperform models trained on small datasets with a lot of features. Google has historically had great success training simple models on large datasets.

Different datasets for different machine learning programs may require wildly different amounts of examples to build a useful model. For some relatively simple problems, a few dozen examples might be sufficient. For other problems, a trillion examples might be insufficient.

It's possible to get good results from a small dataset if you are adapting an existing model already trained on large quantities of data from the same schema.

### Quality and reliability of data

Everyone prefers high quality to low quality, but quality is such a vague concept that it could be defined many different ways. This course defines **quality** pragmatically:

> A high-quality dataset helps your model accomplish its goal. A low quality dataset inhibits your model from accomplishing its goal.
> 

A high-quality dataset is usually also reliable. **Reliability** refers to the degree to which you can *trust* your data. A model trained on a reliable dataset is more likely to yield useful predictions than a model trained on unreliable data.

In *measuring* reliability, you must determine:

- How common are label errors? For example, if your data is labeled by humans, how often did your human raters make mistakes?
- Are your features *noisy*? That is, do the values in your features contain errors? Be realistic—you can't purge your dataset of all noise. Some noise is normal; for example, GPS measurements of any location always fluctuate a little, week to week.
- Is the data properly filtered for your problem? For example, should your dataset include search queries from bots? If you're building a spam-detection system, then likely the answer is yes. However, if you're trying to improve search results for humans, then no.

The following are common causes of unreliable data in datasets:

- Omitted values. For example, a person forgot to enter a value for a house's age.
- Duplicate examples. For example, a server mistakenly uploaded the same log entries twice.
- Bad feature values. For example, someone typed an extra digit, or a thermometer was left out in the sun.
- Bad labels. For example, a person mistakenly labeled a picture of an oak tree as a maple tree.
- Bad sections of data. For example, a certain feature is very reliable, except for that one day when the network kept crashing.

We recommend using automation to flag unreliable data. For example, unit tests that define or rely on an external formal data schema can flag values that fall outside of a defined range.

**Note:** Any sufficiently large or diverse dataset almost certainly contains [**outliers**](https://developers.google.com/machine-learning/glossary#outliers) that fall outside your data schema or unit test bands. Determining how to handle outliers is an important part of machine learning. The [**Numerical data unit**](https://developers.google.com/machine-learning/crash-course/numerical-data) details how to handle numeric outliers.

### Complete vs. incomplete examples

In a perfect world, each example is **complete**; that is, each example contains a value for each feature.

![Figure 1. An example containing values for all five of its
       features.](https://developers.google.com/static/machine-learning/crash-course/images/complete_example.svg)

***Figure 1.** A complete example.*

Unfortunately, real-world examples are often **incomplete**, meaning that at least one feature value is missing.

![Figure 2. An example containing values for four of its five
            features. One feature is marked missing.](https://developers.google.com/static/machine-learning/crash-course/images/incomplete_example.svg)

***Figure 2.** An incomplete example.*

Don't train a model on incomplete examples. Instead, fix or eliminate incomplete examples by doing one of the following:

- Delete incomplete examples.
- [**Impute**](https://developers.google.com/machine-learning/glossary#value-imputation) missing values; that is, convert the incomplete example to a complete example by providing well-reasoned guesses for the missing values.

![Figure 3. A dataset containing three examples, two of which are
            incomplete examples. Someone has stricken these two incomplete
            examples from the dataset.](https://developers.google.com/static/machine-learning/crash-course/images/delete_incomplete_examples.svg)

***Figure 3.** Deleting incomplete examples from the dataset.*

![Figure 4. A dataset containing three examples, two of which were
            incomplete examples containing missing data. Some entity (a human
            or imputation software) has imputed values that replaced the
            missing data.](https://developers.google.com/static/machine-learning/crash-course/images/impute_missing_values.svg)

***Figure 4.** Imputing missing values for incomplete examples.*

If the dataset contains enough complete examples to train a useful model, then consider deleting the incomplete examples. Similarly, if only one feature is missing a significant amount of data and that one feature probably can't help the model much, then consider deleting that feature from the model inputs and seeing how much quality is lost by its removal. If the model works just or almost as well without it, that's great. Conversely, if you don't have enough complete examples to train a useful model, then you might consider imputing missing values.

It's fine to delete useless or redundant examples, but it's bad to delete important examples. Unfortunately, it can be difficult to differentiate between useless and useful examples. If you can't decide whether to delete or impute, consider building two datasets: one formed by deleting incomplete examples and the other by imputing. Then, determine which dataset trains the better model.

Clever algorithms can impute some pretty good missing values; however, imputed values are rarely as good as the actual values. Therefore, a good dataset tells the model which values are imputed and which are actual. One way to do this is to add an extra Boolean column to the dataset that indicates whether a particular feature's value is imputed. For example, given a feature named **`temperature`**, you could add an extra Boolean feature named something like **`temperature_is_imputed`**. Then, during training, the model will probably gradually learn to trust examples containing imputed values for feature **`temperature`** *less* than examples containing actual (non-imputed) values.

Imputation is the process of generating well-reasoned data, not random or deceptive data. Be careful: good imputation can improve your model; bad imputation can hurt your model.

One common algorithm is to use the mean or median as the imputed value. Consequently, when you represent a numerical feature with [**Z-scores**](https://developers.google.com/machine-learning/glossary#z-score-normalization), then the imputed value is typically 0 (because 0 is generally the mean Z-score).

## **Labels**

### Direct versus proxy labels

Consider two different kinds of labels:

- **Direct labels**, which are labels identical to the prediction your model is trying to make. That is, the prediction your model is trying to make is exactly present as a column in your dataset. For example, a column named **`bicycle owner`** would be a direct label for a binary classification model that predicts whether or not a person owns a bicycle.
- **Proxy labels**, which are labels that are similar—but not identical—to the prediction your model is trying to make. For example, a person subscribing to Bicycle Bizarre magazine probably—but not definitely—owns a bicycle.

Direct labels are generally better than proxy labels. If your dataset provides a possible direct label, you should probably use it. Oftentimes though, direct labels aren't available.

Proxy labels are always a compromise—an imperfect approximation of a direct label. However, some proxy labels are close enough approximations to be useful. Models that use proxy labels are only as useful as the connection between the proxy label and the prediction.

Recall that every label must be represented as a floating-point number in the [**feature vector**](https://developers.google.com/machine-learning/glossary#feature-vector) (because machine learning is fundamentally just a huge amalgam of mathematical operations). Sometimes, a direct label exists but can't be easily represented as a floating-point number in the feature vector. In this case, use a proxy label.

### Human-generated data

Some data is **human-generated**; that is, one or more humans examine some information and provide a value, usually for the label. For example, one or more meteorologists could examine pictures of the sky and identify cloud types.

Alternatively, some data is **automatically-generated**. That is, software (possibly, another machine learning model) determines the value. For example, a machine learning model could examine sky pictures and automatically identify cloud types.

This section explores the advantages and disadvantages of human-generated data.

**Advantages**

- Human raters can perform a wide range of tasks that even sophisticated machine learning models may find difficult.
- The process forces the owner of the dataset to develop clear and consistent criteria.

**Disadvantages**

- You typically pay human raters, so human-generated data can be expensive.
- To err is human. Therefore, multiple human raters might have to evaluate the same data.

Think through these questions to determine your needs:

- How skilled must your raters be? (For example, must the raters know a specific language? Do you need linguists for dialogue or NLP applications?)
- How many labeled examples do you need? How soon do you need them?
- What's your budget?

**Always double-check your human raters**. For example, label 1000 examples yourself, and see how your results match other raters' results. If discrepancies surface, don't assume your ratings are the correct ones, especially if a value judgment is involved. If human raters have introduced errors, consider adding instructions to help them and try again.

Looking at your data by hand is a good exercise regardless of how you obtained your data. Andrej Karpathy did this on [ImageNet and wrote about the experience](http://karpathy.github.io/2014/09/02/what-i-learned-from-competing-against-a-convnet-on-imagenet).

Models can train on a mix of automated and human-generated labels. However, for most models, an extra set of human-generated labels (which can become stale) are generally not worth the extra complexity and maintenance. That said, sometimes the human-generated labels can provide extra information not available in the automated labels.

## **Datasets: Class-imbalanced datasets**

### Class-balanced datasets versus class-imbalanced datasets

Consider a dataset containing a [**categorical**](https://developers.google.com/machine-learning/glossary#categorical-data) label whose value is either the positive class or the negative class. In a [**class-balanced dataset**](https://developers.google.com/machine-learning/glossary#class-balanced-dataset), the number of [**positive classes**](https://developers.google.com/machine-learning/glossary#positive-class) and [**negative classes**](https://developers.google.com/machine-learning/glossary#negative-class) is about equal. For example, a dataset containing 235 positive classes and 247 negative classes is a balanced dataset.

In a [**class-imbalanced dataset**](https://developers.google.com/machine-learning/glossary#class-imbalanced-dataset), one label is considerably more common than the other. In the real world, class-imbalanced datasets are far more common than class-balanced datasets. For example, in a dataset of credit card transactions, fraudulent purchases might make up less than 0.1% of the examples. Similarly, in a medical diagnosis dataset, the number of patients with a rare virus might be less than 0.01% of the total examples. In a class-imbalanced dataset:

- The *more* common label is called the [**majority class**](https://developers.google.com/machine-learning/glossary#majority_class).
- The *less* common label is called the [**minority class**](https://developers.google.com/machine-learning/glossary#minority_class).

### The difficulty of training severely class-imbalanced datasets

Training aims to create a model that successfully distinguishes the positive class from the negative class. To do so, [**batches**](https://developers.google.com/machine-learning/glossary#batch) need a sufficient number of *both* positive classes and negative classes. That's not a problem when training on a mildly class-imbalanced dataset since even small batches typically contain sufficient examples of both the positive class and the negative class. However, a severely class-imbalanced dataset might not contain enough minority class examples for proper training.

For example, consider the class-imbalanced dataset illustrated in Figure 6 in which:

- 200 labels are in the majority class.
- 2 labels are in the minority class.

![FloralDataset200Sunflowers2Roses.png](Numerical%20and%20Categorical%20Data,%20and%20Generalization/FloralDataset200Sunflowers2Roses.png)

***Figure 6.** A highly imbalanced floral dataset containing far more sunflowers than roses.*

If the batch size is 20, most batches won't contain any examples of the minority class. If the batch size is 100, each batch will contain an average of only one minority class example, which is insufficient for proper training. Even a much larger batch size will still yield such an imbalanced proportion that the model might not train properly.

**Note:** [**Accuracy**](https://developers.google.com/machine-learning/glossary#accuracy) is usually a poor metric for assessing a model trained on a class-imbalanced dataset. See [Classification: Accuracy, recall, precision, and related metrics](https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall) for details.

### Training a class-imbalanced dataset

During training, a model should learn two things:

- What each class looks like; that is, what feature values correspond to what class?
- How common each class is; that is, what is the relative distribution of the classes?

Standard training conflates these two goals. In contrast, the following two-step technique called **downsampling and upweighting the majority class** separates these two goals, enabling the model to achieve *both* goals.

**Note:** Many students read the following section and say some variant of, "That just can't be right." Be warned that downsampling and upweighting the majority class is somewhat counterintuitive.

### Step 1: Downsample the majority class

[**Downsampling**](https://developers.google.com/machine-learning/glossary#downsampling) means training on a disproportionately low percentage of majority class examples. That is, you artificially force a class-imbalanced dataset to become somewhat more balanced by omitting many of the majority class examples from training. Downsampling greatly increases the probability that each batch contains enough examples of the minority class to train the model properly and efficiently.

For example, the class-imbalanced dataset shown in Figure 6 consists of 99% majority class and 1% minority class examples. Downsampling the majority class by a factor of 25 artificially creates a more balanced training set (80% majority class to 20% minority class) suggested in Figure 7:

![FloralDatasetDownsampling.png](Numerical%20and%20Categorical%20Data,%20and%20Generalization/FloralDatasetDownsampling.png)

***Figure 7.** Downsampling the majority class by a factor of 25.*

### Step 2: Upweight the downsampled class

Downsampling introduces a [**prediction bias**](https://developers.google.com/machine-learning/glossary#prediction-bias) by showing the model an artificial world where the classes are more balanced than in the real world. To correct this bias, you must "upweight" the majority classes by the factor to which you downsampled. Upweighting means treating the loss on a majority class example more harshly than the loss on a minority class example.

For example, we downsampled the majority class by a factor of 25, so we must upweight the majority class by a factor of 25. That is, when the model mistakenly predicts the majority class, treat the loss as if it were 25 errors (multiply the regular loss by 25).

![FloralDatasetUpweighting.png](Numerical%20and%20Categorical%20Data,%20and%20Generalization/FloralDatasetUpweighting.png)

***Figure 8.** Upweighting the majority class by a factor of 25.*

How much should you downsample and upweight to rebalance your dataset? To determine the answer, you should experiment with different downsampling and upweighting factors just as you would experiment with other [**hyperparameters**](https://developers.google.com/machine-learning/glossary#hyperparameter).

### Benefits of this technique

Downsampling and upweighting the majority class brings the following benefits:

- **Better model:** The resultant model "knows" both of the following:
    - The connection between features and labels
    - The true distribution of the classes
- **Faster convergence:** During training, the model sees the minority class more often, which helps the model converge faster.

## **Dividing the original dataset**

All good software engineering projects devote considerable energy to *testing* their apps. Similarly, we strongly recommend testing your ML model to determine the correctness of its predictions.

### Training, validation, and test sets

You should test a model against a *different* set of examples than those used to train the model. As you'll learn [a little later](https://developers.google.com/machine-learning/crash-course/overfitting/dividing-datasets?_gl=1*1v3hw0f*_up*MQ..*_ga*MTc0MDI1NDQ2Ni4xNzU5MzQwOTg2*_ga_SM8HXJ53K2*czE3NTkzNDA5ODYkbzEkZzAkdDE3NTkzNDA5ODYkajYwJGwwJGgw#additional_problems_with_test_sets), testing on different examples is stronger proof of your model's fitness than testing on the same set of examples. Where do you get those different examples? Traditionally in machine learning, you get those different examples by splitting the original dataset. You might assume, therefore, that you should split the original dataset into two subsets:

- A [**training set**](https://developers.google.com/machine-learning/glossary#training-set) that the model trains on.
- A [**test set**](https://developers.google.com/machine-learning/glossary#test-set) for evaluation of the trained model.

![Figure 8. A horizontal bar divided into two pieces: ~80% of which
            is the training set and ~20% is the test set.](https://developers.google.com/static/machine-learning/crash-course/images/PartitionTwoSets.png)

***Figure 8.** Not an optimal split.*

Dividing the dataset into two sets is a decent idea, but a better approach is to divide the dataset into *three* subsets. In addition to the training set and the test set, the third subset is:

- A [**validation set**](https://developers.google.com/machine-learning/glossary#validation-set) performs the initial testing on the model as it is being trained.

![Figure 9. A horizontal bar divided into three pieces: 70% of which
            is the training set, 15% the validation set, and 15%
            the test set](https://developers.google.com/static/machine-learning/crash-course/images/PartitionThreeSets.png)

***Figure 9.** A much better split.*

Use the **validation set** to evaluate results from the training set. After repeated use of the validation set suggests that your model is making good predictions, use the test set to double-check your model.

The following figure suggests this workflow. In the figure, "Tweak model" means adjusting anything about the model —from changing the learning rate, to adding or removing features, to designing a completely new model from scratch.

![Figure 10. A workflow diagram consisting of the following stages:
            1. Train model on the training set.
            2. Evaluate model on the validation set.
            3. Tweak model according to results on the validation set.
            4. Iterate on 1, 2, and 3, ultimately picking the model that does
               best on the validation set.
            5. Confirm the results on the test set.](https://developers.google.com/static/machine-learning/crash-course/images/workflow_with_validation_set.svg)

***Figure 10.** A good workflow for development and testing.*

**Note:** When you transform a feature in your training set, you must make the *same* transformation in the validation set, test set, and real-world dataset.

The workflow shown in Figure 10 is optimal, but even with that workflow, test sets and validation sets still "wear out" with repeated use. That is, the more you use the same data to make decisions about hyperparameter settings or other model improvements, the less confidence that the model will make good predictions on new data. For this reason, it's a good idea to collect more data to "refresh" the test set and validation set. Starting anew is a great reset.

### Additional problems with test sets

As the previous question illustrates, duplicate examples can affect model evaluation. After splitting a dataset into training, validation, and test sets, delete any examples in the validation set or test set that are duplicates of examples in the training set. The only fair test of a model is against new examples, not duplicates.

For example, consider a model that predicts whether an email is spam, using the subject line, email body, and sender's email address as features. Suppose you divide the data into training and test sets, with an 80-20 split. After training, the model achieves 99% precision on both the training set and the test set. You'd probably expect a lower precision on the test set, so you take another look at the data and discover that many of the examples in the test set are duplicates of examples in the training set. The problem is that you neglected to scrub duplicate entries for the same spam email from your input database before splitting the data. You've inadvertently trained on some of your test data.

In summary, a good test set or validation set meets all of the following criteria:

- Large enough to yield statistically significant testing results.
- Representative of the dataset as a whole. In other words, don't pick a test set with different characteristics than the training set.
- Representative of the real-world data that the model will encounter as part of its business purpose.
- Zero examples duplicated in the training set.

## **Transforming data**

Machine learning models can only train on floating-point values. However, many dataset features are *not* naturally floating-point values. Therefore, one important part of machine learning is transforming non-floating-point features to floating-point representations.

For example, suppose **`street names`** is a feature. Most street names are strings, such as "Broadway" or "Vilakazi". Your model can't train on "Broadway", so you must transform "Broadway" to a floating-point number. The [Categorical Data module](https://developers.google.com/machine-learning/crash-course/categorical-data) explains how to do this.

Additionally, you should even transform most floating-point features. This transformation process, called [**normalization**](https://developers.google.com/machine-learning/glossary#normalization), converts floating-point numbers to a constrained range that improves model training. The [Numerical Data module](https://developers.google.com/machine-learning/crash-course/numerical-data) explains how to do this.

### Sample data when you have too much of it

Some organizations are blessed with an abundance of data. When the dataset contains too many examples, you must select a *subset* of examples for training. When possible, select the subset that is most relevant to your model's predictions.

### Filter examples containing PII

Good datasets omit examples containing Personally Identifiable Information (PII). This policy helps safeguard privacy but can influence the model.

See the Safety and Privacy module later in the course for more on these topics.

## Generalization

Generalization is a problem where ML models trains on a small subset of the real world application data, meaning that it overfits the data as it matches this subset very well. To solve this you need to broaden the data and ensure you cover all contexts.

## **Overfitting**

[**Overfitting**](https://developers.google.com/machine-learning/glossary#overfitting) means creating a model that matches (*memorizes*) the [**training set**](https://developers.google.com/machine-learning/glossary#training-set) so closely that the model fails to make correct predictions on new data. An overfit model is analogous to an invention that performs well in the lab but is worthless in the real world.

**Tip:** Overfitting is a common problem in machine learning, not an academic hypothetical.

In Figure 11, imagine that each geometric shape represents a tree's position in a square forest. The blue diamonds mark the locations of healthy trees, while the orange circles mark the locations of sick trees.

![Figure 11. This figure contains about 60 dots, half of which are
            healthy trees and the other half sick trees.
            The healthy trees are mainly in the northeast quadrant, though a few
            healthy trees sneak into the northwest quadrants. The sick trees
            are mainly in the southeast quadrant, but a few of the sick trees
            spill into other quadrants.](https://developers.google.com/static/machine-learning/crash-course/images/TreesTrainingSet.svg)

***Figure 11.** Training set: locations of healthy and sick trees in a square forest.*

Mentally draw any shapes—lines, curves, ovals...anything—to separate the healthy trees from the sick trees. Then, expand the next line to examine one possible separation.

**Expand to see one possible solution (Figure 12).**

![Figure 12. The same arrangement of healthy and sick trees as in
            Figure 11. However, a model of complex geometric shapes separates
            nearly all of the healthy trees from the sick trees.](https://developers.google.com/static/machine-learning/crash-course/images/TreesTrainingSetComplexModel.svg)

***Figure 12.** A complex model for distinguishing sick from healthy trees.*

The complex shapes shown in Figure 12 successfully categorized all but two of the trees. If we think of the shapes as a model, then this is a fantastic model.

Or is it? A truly excellent model successfully categorizes *new* examples. Figure 13 shows what happens when that same model makes predictions on new examples from the test set:

![Figure 13. A new batch of healthy and sick trees overlaid on the
            model shown in Figure 12. The model miscategorizes many of the
            trees.](https://developers.google.com/static/machine-learning/crash-course/images/TreesTestSetComplexModel.svg)

***Figure 13.**Test set: a complex model for distinguishing sick from healthy trees.*

So, the complex model shown in Figure 12 did a great job on the training set but a pretty bad job on the test set. This is a classic case of a model *overfitting* to the training set data.

### Fitting, overfitting, and underfitting

A model must make good predictions on *new* data. That is, you're aiming to create a model that "fits" new data.

As you've seen, an overfit model makes excellent predictions on the training set but poor predictions on new data. An [**underfit**](https://developers.google.com/machine-learning/glossary#underfitting) model doesn't even make good predictions on the training data. If an overfit model is like a product that performs well in the lab but poorly in the real world, then an underfit model is like a product that doesn't even do well in the lab.

![Figure 14. Cartesian plot. X-axis is labeled 'quality of predictions
            on training set.' Y-axis is labeled 'quality of predictions on
            real-world data.' A curve starts at the origin and rises gradually,
            but then falls just as quickly. The lower-left portion of the curve
            (low quality of predictions on real-world data and low quality of
            predictions on training set) is labeled 'underfit models.' The
            lower-right portion of the curve (low quality of predictions on
            real-world data but high quality of predictions on training set)
            is labeled 'overfit models.' The peak of the curve (high quality
            of predictions on real-world data and medium quality of predictions
            on training set) is labeled 'fit models.'](https://developers.google.com/static/machine-learning/crash-course/images/underfit_fit_overfit.svg)

***Figure 14.** Underfit, fit, and overfit models.*

[**Generalization**](https://developers.google.com/machine-learning/glossary#generalization) is the opposite of overfitting. That is, a model that *generalizes well* makes good predictions on new data. Your goal is to create a model that generalizes well to new data.

### Detecting overfitting

The following curves help you detect overfitting:

- loss curves
- generalization curves

A [**loss curve**](https://developers.google.com/machine-learning/glossary#loss-curve) plots a model's loss against the number of training iterations. A graph that shows two or more loss curves is called a [**generalization curve**](https://developers.google.com/machine-learning/glossary#generalization-curve). The following generalization curve shows two loss curves:

![Figure 15. The loss function for the training set gradually
            declines. The loss function for the validation set also declines,
            but then it starts to rise after a certain number of iterations.](https://developers.google.com/static/machine-learning/crash-course/images/RegularizationTwoLossFunctions.png)

***Figure 15.** A generalization curve that strongly implies overfitting.*

Notice that the two loss curves behave similarly at first and then diverge. That is, after a certain number of iterations, loss declines or holds steady (converges) for the training set, but increases for the validation set. This suggests overfitting.

In contrast, a generalization curve for a well-fit model shows two loss curves that have similar shapes.

### What causes overfitting?

Very broadly speaking, overfitting is caused by one or both of the following problems:

- The training set doesn't adequately represent real life data (or the validation set or test set).
- The model is too complex.

### Generalization conditions

A model trains on a training set, but the real test of a model's worth is how well it makes predictions on new examples, particularly on real-world data. While developing a model, your test set serves as a proxy for real-world data. Training a model that generalizes well implies the following dataset conditions:

- Examples must be [**independently and identically distributed**](https://developers.google.com/machine-learning/glossary#independently-and-identically-distributed-i.i.d), which is a fancy way of saying that your examples can't influence each other.
- The dataset is [**stationary**](https://developers.google.com/machine-learning/glossary#stationarity), meaning the dataset doesn't change significantly over time.
- The dataset partitions have the same distribution. That is, the examples in the training set are statistically similar to the examples in the validation set, test set, and real-world data.

Explore the preceding conditions through the following exercises.

## **Overfitting: Model complexity**

The previous unit introduced the following model, which miscategorized a lot of trees in the test set:

![Figure 16. The same image as Figure 13. This is a complex shape that
            miscategorizes many trees.](https://developers.google.com/static/machine-learning/crash-course/images/TreesTestSetComplexModel.svg)

***Figure 16.** The misbehaving complex model from the previous unit.*

The preceding model contains a lot of complex shapes. Would a simpler model handle new data better? Suppose you replace the complex model with a ridiculously simple model--a straight line.

![Figure 17. A straight line model that does an excellent job
            separating the sick trees from the healthy trees.](https://developers.google.com/static/machine-learning/crash-course/images/TreesTestSetSimpleModel.svg)

***Figure 17.** A much simpler model.*

The simple model generalizes better than the complex model on new data. That is, the simple model made better predictions on the test set than the complex model.

Simplicity has been beating complexity for a long time. In fact, the preference for simplicity dates back to ancient Greece. Centuries later, a fourteenth-century friar named William of Occam formalized the preference for simplicity in a philosophy known as [Occam's razor](https://wikipedia.org/wiki/Occam%27s_razor). This philosophy remains an essential underlying principle of many sciences, including machine learning.

**Note:** Complex models typically outperform simple models on the training set. However, simple models typically outperform complex models on the test set (which is more important).

### Regularization

Machine learning models must simultaneously meet two conflicting goals:

- Fit data well.
- Fit data as simply as possible.

One approach to keeping a model simple is to penalize complex models; that is, to force the model to become simpler during training. Penalizing complex models is one form of **regularization**.

**A regularization analogy:** Suppose every student in a lecture hall had a little buzzer that emitted a sound that annoyed the professor. Students would press the buzzer whenever the professor's lecture became too complicated. Annoyed, the professor would be forced to simplify the lecture. The professor would complain, "When I simplify, I'm not being precise enough." The students would counter with, "The only goal is to explain it simply enough that I understand it." Gradually, the buzzers would train the professor to give an appropriately simple lecture, even if the simpler lecture isn't as sufficiently precise.

### Loss and complexity

So far, this course has suggested that the only goal when training was to minimize loss; that is:

- minimize(loss)

As you've seen, models focused solely on minimizing loss tend to overfit. A better training optimization algorithm minimizes some combination of loss and complexity:

- minimize(loss + complexity)

Unfortunately, loss and complexity are typically inversely related. As complexity increases, loss decreases. As complexity decreases, loss increases. You should find a reasonable middle ground where the model makes good predictions on both the training data and real-world data. That is, your model should find a reasonable compromise between loss and complexity.

## **Overfitting: L2 regularization**

[**L2 regularization**](https://developers.google.com/machine-learning/glossary#l2-regularization) is a popular regularization metric, which uses the following formula:

- $L_2 \ regularization = w^2_1+w^2_2+...+w^2_n$

For example, the following table shows the calculation of L2 regularization for a model with six weights:

|  | **Value** | **Squared value** |
| --- | --- | --- |
| w1 | 0.2 | 0.04 |
| w2 | -0.5 | 0.25 |
| w3 | 5.0 | 25.0 |
| w4 | -1.2 | 1.44 |
| w5 | 0.3 | 0.09 |
| w6 | -0.1 | 0.01 |
|  |  | **26.83** = total |

Notice that weights close to zero don't affect L2 regularization much, but large weights can have a huge impact. For example, in the preceding calculation:

- A single weight ($w_3$) contributes about 93% of the total complexity.
- The other five weights collectively contribute only about 7% of the total complexity.

L2 regularization encourages weights *toward* 0, but never pushes weights all the way to zero.

### Regularization rate (lambda)

As noted, training attempts to minimize some combination of loss and complexity:

- $minimize(loss+complexity)$

Model developers tune the overall impact of complexity on model training by multiplying its value by a scalar called the [**regularization rate**](https://developers.google.com/machine-learning/glossary#regularization-rate). The Greek character lambda typically symbolizes the regularization rate.

That is, model developers aim to do the following:

- $minimize(loss+λ \ complexity)$

A high regularization rate:

- Strengthens the influence of regularization, thereby reducing the chances of overfitting.
- Tends to produce a histogram of model weights having the following characteristics:
    - a normal distribution
    - a mean weight of 0.

A low regularization rate:

- Lowers the influence of regularization, thereby increasing the chances of overfitting.
- Tends to produce a histogram of model weights with a flat distribution.

For example, the histogram of model weights for a high regularization rate might look as shown in Figure 18.

![Figure 18. Histogram of a model's weights with a mean of zero and
            a normal distribution.](https://developers.google.com/static/machine-learning/crash-course/images/HighLambda.svg)

***Figure 18.** Weight histogram for a high regularization rate. Mean is zero. Normal distribution.*

In contrast, a low regularization rate tends to yield a flatter histogram, as shown in Figure 19.

![Figure 19. Histogram of a model's weights with a mean of zero that
            is somewhere between a flat distribution and a normal
            distribution.](https://developers.google.com/static/machine-learning/crash-course/images/LowLambda.svg)

***Figure 19.** Weight histogram for a low regularization rate. Mean may or may not be zero.*

**Note:** Setting the regularization rate to zero removes regularization completely. In this case, training focuses exclusively on minimizing loss, which poses the highest possible overfitting risk.

### Picking the regularization rate

The ideal regularization rate produces a model that generalizes well to new, previously unseen data. Unfortunately, that ideal value is data-dependent, so you must do some tuning.

### Early stopping: an alternative to complexity-based regularization

[**Early stopping**](https://developers.google.com/machine-learning/glossary#early-stopping) is a regularization method that doesn't involve a calculation of complexity. Instead, early stopping simply means ending training before the model fully converges. For example, you end training when the loss curve for the validation set starts to increase (slope becomes positive).

Although early stopping usually increases training loss, it can decrease test loss.

Early stopping is a quick, but rarely optimal, form of regularization. The resulting model is very unlikely to be as good as a model trained thoroughly on the ideal regularization rate.

### Finding equilibrium between learning rate and regularization rate

[**Learning rate**](https://developers.google.com/machine-learning/glossary#learning-rate) and regularization rate tend to pull weights in opposite directions. A high learning rate often pulls weights *away from* zero; a high regularization rate pulls weights *towards* zero.

If the regularization rate is high with respect to the learning rate, the weak weights tend to produce a model that makes poor predictions. Conversely, if the learning rate is high with respect to the regularization rate, the strong weights tend to produce an overfit model.

Your goal is to find the equilibrium between learning rate and regularization rate. This can be challenging. Worst of all, once you find that elusive balance, you may have to ultimately change the learning rate. And, when you change the learning rate, you'll again have to find the ideal regularization rate.

## **Overfitting: Interpreting loss curves**