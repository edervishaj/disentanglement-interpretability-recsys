# Are Representation Disentanglement and Interpretability Linked in Recommendation Models? A Critical Review and Reproducibility Study

This repository contains the code for reproducing the experiments of the paper "Are Representation Disentanglement and Interpretability Linked in Recommendation Models? A Critical Review and Reproducibility Study" by Ervin Dervishaj, Tuukka Ruotsalo, Maria Maistro and Christina Lioma, accepted at ECIR 2025.

## How to use this repo

In order to run the code and experiments, first setup a `Conda` environment with `Python 3.9.x` and install the following packages:

```shell
conda create -n py39 python=3.9 numpy pandas scipy matplotlib requests hyperopt tqdm seaborn requests cython pyyaml cxx-compiler lime pingouin shap=0.44 scikit-learn=1.0.2
```

Install also these packages through `pip`:
```shell
conda activate py39
pip install recpack==0.3.5 tensorflow-probability tf-keras
```

Since some part of the code are written in Cython (`cymetrics.pyx`), you need to first compile it:

```shell
sh compile_cython.py
```

The code is structured as follows:

* `/datasets`: the code for our datasets.
* `/models`: the code for the reproduced models.
* `/ds_utils`: code for creating/loading the datasets.
* `/main.py`: main entry point of our experiments.
* `/configurations.py`: code that defines the hyperparameters ranges for tuning each model.
* `/metrics.py`: the code that implements the disentanglement and interpretability metrics.
* `/utils.py`: utilities used by the above files.
* `/experiments`: folder where all experimental results are collected.
* `/experiments/datasets`: folder where dataset files are collected.
* `/experiments/evaluations`: folder where evaluation results are collected.
* `/experiments/param-search`: folder where hyperparameter tuning results are collected.

## Preparing the dataset(s)

To prepare one of the datasets, use the following command:

```shell
python main.py prep-dataset <dataset> --min-user-ratings <user-ratings> --min-item-ratings <item-ratings> --min-rating-binarize <rating-binarize> --test-fraction <test-fraction> --validation-fraction <validation-fraction> --seed <seed>
```

where:

* `<dataset>` is a value between `AmazonCD`, `ML1M`, `Yelp`, `GoodReadsChildren`.
* `<user-ratings>` and `<item-ratings>` define the k-core for filtering users and items.
* `<rating-binarize>` specifies the binarization for explicit ratings (values >=`<rating-binarize>` will be set to 1, everything else to 0).
* `<test-fraction>` defines the fraction of the interactions that will be used as test set.
* `<validation-fraction>` defines the fraction of the interactions that will be used as validation set.
* `<seed>` is the randomization seed between `41`, `53`, `61`, `67`, `71`.


## Tuning the model(s)

To tune one of the models, use the following command:

```shell
python main.py param-search <model> <dataset> --K <K> --evals <evals> --seed <seed> 
```

where:

* `<model>` is a value between `toppop`, `puresvd`, `multidae`, `multivae`, `betavae`, `macridvae`.
* `<dataset>` is a value between `AmazonCD`, `ML1M`, `Yelp`, `GoodReadsChildren`.
* `K` is the cutoff for the evaluation during the Bayesian Search Optimization. Default value is 100 (in our experiments we tune for NDCG@100).
* `<evals>` is the number parameter searches of Bayesian Search Optimization.
* `<seed>` is the randomization seed between `41`, `53`, `61`, `67`, `71`.

## Evaluating a tuned model

To evaluate a previously tuned model, use the following command:

```shell
python main.py eval <model> <dataset> --metrics <metrics> --K <K> --evals <evals> --seed <seed>
```

where:

* `<model>` is a value between `toppop`, `puresvd`, `multidae`, `multivae`, `betavae`, `macridvae`.
* `<dataset>` is a value between `AmazonCD`, `ML1M`, `Yelp`, `GoodReadsChildren`.
* `<metrics>` is a value between `ndcg`, `recall`, `mrr`, `coverage`, `disentanglement`, `completeness`, `lime`, `shap`.
* `K` is the cutoff for the evaluation during the Bayesian Search Optimization. Default value is 100 (in our experiments we tune for NDCG@100).
* `<evals>` is the number parameter searches of Bayesian Search Optimization.
* `<seed>` is the randomization seed between `41`, `53`, `61`, `67`, `71`.

## Correlation Analysis

To plot the "Repeated Measurements Correlation" analysis run the Jupyter Notebook file `/correlations.ipynb`.

## Best hyperparameters for each model-dataset-seed

<table>
    <tr>
        <th>Dataset</th>
        <th>Model</th>
        <th>Seed</th>
        <th>Epochs</th>
        <th>Batch Size</th>
        <th>Learning Rate</th>
        <th>Latent Dimensions</th>
        <th>Regularization</th>
        <th>Num. hidden layers</th>
        <th>Beta</th>
        <th>Std</th>
    </tr>
    <tr>
        <td>AmazonCD</td>
        <td>puresvd</td>
        <td>41</td>
        <td>0</td>
        <td>128</td>
        <td>0.1</td>
        <td>20.0</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>AmazonCD</td>
        <td>puresvd</td>
        <td>53</td>
        <td>0</td>
        <td>128</td>
        <td>0.1</td>
        <td>20.0</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>AmazonCD</td>
        <td>puresvd</td>
        <td>61</td>
        <td>0</td>
        <td>128</td>
        <td>0.1</td>
        <td>20.0</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>AmazonCD</td>
        <td>puresvd</td>
        <td>67</td>
        <td>0</td>
        <td>128</td>
        <td>0.1</td>
        <td>20.0</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>AmazonCD</td>
        <td>puresvd</td>
        <td>71</td>
        <td>0</td>
        <td>128</td>
        <td>0.1</td>
        <td>20.0</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>AmazonCD</td>
        <td>multidae</td>
        <td>41</td>
        <td>90</td>
        <td>512</td>
        <td>0.0036475252037085687</td>
        <td>16.0</td>
        <td>0.0008773963359516143</td>
        <td>1.0</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>AmazonCD</td>
        <td>multidae</td>
        <td>53</td>
        <td>305</td>
        <td>128</td>
        <td>0.00020605852122304832</td>
        <td>16.0</td>
        <td>0.00040291999718222493</td>
        <td>1.0</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>AmazonCD</td>
        <td>multidae</td>
        <td>61</td>
        <td>205</td>
        <td>1024</td>
        <td>0.001367425527500579</td>
        <td>14.0</td>
        <td>0.00029092922643997034</td>
        <td>1.0</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>AmazonCD</td>
        <td>multidae</td>
        <td>67</td>
        <td>125</td>
        <td>512</td>
        <td>0.0006151244180088173</td>
        <td>18.0</td>
        <td>0.00021732289852938457</td>
        <td>2.0</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>AmazonCD</td>
        <td>multidae</td>
        <td>71</td>
        <td>205</td>
        <td>128</td>
        <td>0.001518658286442731</td>
        <td>19.0</td>
        <td>0.00019421012388604438</td>
        <td>0.0</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>AmazonCD</td>
        <td>multivae</td>
        <td>41</td>
        <td>105</td>
        <td>1024</td>
        <td>0.003903899555925899</td>
        <td>17.0</td>
        <td>7.861864043967243e-05</td>
        <td>0.0</td>
        <td>0.4</td>
        <td>-</td>
    </tr>
    <tr>
        <td>AmazonCD</td>
        <td>multivae</td>
        <td>53</td>
        <td>215</td>
        <td>1024</td>
        <td>0.001998590526491362</td>
        <td>20.0</td>
        <td>0.00015971662484819823</td>
        <td>0.0</td>
        <td>0.01</td>
        <td>-</td>
    </tr>
    <tr>
        <td>AmazonCD</td>
        <td>multivae</td>
        <td>61</td>
        <td>165</td>
        <td>256</td>
        <td>0.001486619795137357</td>
        <td>20.0</td>
        <td>0.00038884666484368416</td>
        <td>1.0</td>
        <td>0.09</td>
        <td>-</td>
    </tr>
    <tr>
        <td>AmazonCD</td>
        <td>multivae</td>
        <td>67</td>
        <td>135</td>
        <td>128</td>
        <td>0.003557688981494811</td>
        <td>20.0</td>
        <td>7.504142713107713e-05</td>
        <td>0.0</td>
        <td>0.62</td>
        <td>-</td>
    </tr>
    <tr>
        <td>AmazonCD</td>
        <td>multivae</td>
        <td>71</td>
        <td>85</td>
        <td>1024</td>
        <td>0.010285966392682075</td>
        <td>20.0</td>
        <td>0.0002760722720333667</td>
        <td>0.0</td>
        <td>0.05</td>
        <td>-</td>
    </tr>
    <tr>
        <td>AmazonCD</td>
        <td>betavae</td>
        <td>41</td>
        <td>160</td>
        <td>256</td>
        <td>0.0022168412987808215</td>
        <td>16.0</td>
        <td>4.829278226883074e-05</td>
        <td>0.0</td>
        <td>1.3809439700885209</td>
        <td>-</td>
    </tr>
    <tr>
        <td>AmazonCD</td>
        <td>betavae</td>
        <td>53</td>
        <td>55</td>
        <td>256</td>
        <td>0.0017674007890127277</td>
        <td>20.0</td>
        <td>1.6202543150033716e-05</td>
        <td>1.0</td>
        <td>1.1609300581688546</td>
        <td>-</td>
    </tr>
    <tr>
        <td>AmazonCD</td>
        <td>betavae</td>
        <td>61</td>
        <td>100</td>
        <td>512</td>
        <td>0.0004797016658975882</td>
        <td>20.0</td>
        <td>1.73435794505457e-05</td>
        <td>4.0</td>
        <td>1.1453639637775692</td>
        <td>-</td>
    </tr>
    <tr>
        <td>AmazonCD</td>
        <td>betavae</td>
        <td>67</td>
        <td>125</td>
        <td>1024</td>
        <td>0.003987460710023986</td>
        <td>6.0</td>
        <td>0.0004541227899269731</td>
        <td>1.0</td>
        <td>2.1797827323823804</td>
        <td>-</td>
    </tr>
    <tr>
        <td>AmazonCD</td>
        <td>betavae</td>
        <td>71</td>
        <td>380</td>
        <td>128</td>
        <td>0.0004513202970614056</td>
        <td>10.0</td>
        <td>6.547052897644267e-06</td>
        <td>0.0</td>
        <td>1.1320617514955664</td>
        <td>-</td>
    </tr>
    <tr>
        <td>AmazonCD</td>
        <td>macridvae</td>
        <td>41</td>
        <td>340</td>
        <td>1024</td>
        <td>7.353999724410027e-05</td>
        <td>13.0</td>
        <td>1.3745826992826976e-05</td>
        <td>-</td>
        <td>4.3500000000000005</td>
        <td>0.07500000000000001</td>
    </tr>
    <tr>
        <td>AmazonCD</td>
        <td>macridvae</td>
        <td>53</td>
        <td>180</td>
        <td>256</td>
        <td>7.707493148321149e-05</td>
        <td>13.0</td>
        <td>2.5269759362269575e-05</td>
        <td>-</td>
        <td>4.95</td>
        <td>0.05</td>
    </tr>
    <tr>
        <td>AmazonCD</td>
        <td>macridvae</td>
        <td>61</td>
        <td>215</td>
        <td>512</td>
        <td>0.00010426007772493048</td>
        <td>13.0</td>
        <td>0.00030346444658303496</td>
        <td>-</td>
        <td>3.0500000000000003</td>
        <td>0.07500000000000001</td>
    </tr>
    <tr>
        <td>AmazonCD</td>
        <td>macridvae</td>
        <td>67</td>
        <td>115</td>
        <td>512</td>
        <td>0.0001542680031247968</td>
        <td>7.0</td>
        <td>0.0055561628506111335</td>
        <td>-</td>
        <td>1.85</td>
        <td>0.07500000000000001</td>
    </tr>
    <tr>
        <td>AmazonCD</td>
        <td>macridvae</td>
        <td>71</td>
        <td>105</td>
        <td>128</td>
        <td>0.00014700952856565834</td>
        <td>20.0</td>
        <td>0.0005230520706332012</td>
        <td>-</td>
        <td>2.0</td>
        <td>0.07500000000000001</td>
    </tr>
    <tr>
        <td>ML1M</td>
        <td>puresvd</td>
        <td>41</td>
        <td>0</td>
        <td>128</td>
        <td>0.1</td>
        <td>20.0</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>ML1M</td>
        <td>puresvd</td>
        <td>53</td>
        <td>0</td>
        <td>128</td>
        <td>0.1</td>
        <td>20.0</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>ML1M</td>
        <td>puresvd</td>
        <td>61</td>
        <td>0</td>
        <td>128</td>
        <td>0.1</td>
        <td>20.0</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>ML1M</td>
        <td>puresvd</td>
        <td>67</td>
        <td>0</td>
        <td>128</td>
        <td>0.1</td>
        <td>20.0</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>ML1M</td>
        <td>puresvd</td>
        <td>71</td>
        <td>0</td>
        <td>128</td>
        <td>0.1</td>
        <td>20.0</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>ML1M</td>
        <td>multidae</td>
        <td>41</td>
        <td>190</td>
        <td>1024</td>
        <td>0.0017612519080986962</td>
        <td>15.0</td>
        <td>0.00100334964986017</td>
        <td>1.0</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>ML1M</td>
        <td>multidae</td>
        <td>53</td>
        <td>190</td>
        <td>128</td>
        <td>0.0005374965093172599</td>
        <td>16.0</td>
        <td>0.001563042521154859</td>
        <td>1.0</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>ML1M</td>
        <td>multidae</td>
        <td>61</td>
        <td>165</td>
        <td>128</td>
        <td>0.0040082591494768555</td>
        <td>19.0</td>
        <td>0.0007663643083097882</td>
        <td>0.0</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>ML1M</td>
        <td>multidae</td>
        <td>67</td>
        <td>355</td>
        <td>512</td>
        <td>0.0033205421586409455</td>
        <td>20.0</td>
        <td>0.00047036393937287913</td>
        <td>0.0</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>ML1M</td>
        <td>multidae</td>
        <td>71</td>
        <td>160</td>
        <td>1024</td>
        <td>0.002054265523265352</td>
        <td>19.0</td>
        <td>0.0005698972640245481</td>
        <td>1.0</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>ML1M</td>
        <td>multivae</td>
        <td>41</td>
        <td>365</td>
        <td>512</td>
        <td>0.0007358570741486225</td>
        <td>19.0</td>
        <td>0.0007578398464872918</td>
        <td>0.0</td>
        <td>0.6900000000000001</td>
        <td>-</td>
    </tr>
    <tr>
        <td>ML1M</td>
        <td>multivae</td>
        <td>53</td>
        <td>80</td>
        <td>512</td>
        <td>0.04911603238657355</td>
        <td>20.0</td>
        <td>0.0007187719542882984</td>
        <td>0.0</td>
        <td>0.97</td>
        <td>-</td>
    </tr>
    <tr>
        <td>ML1M</td>
        <td>multivae</td>
        <td>61</td>
        <td>285</td>
        <td>1024</td>
        <td>0.0018227395755941508</td>
        <td>20.0</td>
        <td>0.00021500643724408136</td>
        <td>0.0</td>
        <td>0.16</td>
        <td>-</td>
    </tr>
    <tr>
        <td>ML1M</td>
        <td>multivae</td>
        <td>67</td>
        <td>190</td>
        <td>512</td>
        <td>0.002761840628452123</td>
        <td>20.0</td>
        <td>4.827465573345749e-05</td>
        <td>0.0</td>
        <td>1.0</td>
        <td>-</td>
    </tr>
    <tr>
        <td>ML1M</td>
        <td>multivae</td>
        <td>71</td>
        <td>155</td>
        <td>1024</td>
        <td>0.007387783193721818</td>
        <td>19.0</td>
        <td>5.454217800969518e-05</td>
        <td>0.0</td>
        <td>0.45</td>
        <td>-</td>
    </tr>
    <tr>
        <td>ML1M</td>
        <td>betavae</td>
        <td>41</td>
        <td>90</td>
        <td>256</td>
        <td>0.001980007144711448</td>
        <td>9.0</td>
        <td>0.0005748032323251092</td>
        <td>1.0</td>
        <td>1.7267817897455067</td>
        <td>-</td>
    </tr>
    <tr>
        <td>ML1M</td>
        <td>betavae</td>
        <td>53</td>
        <td>50</td>
        <td>512</td>
        <td>0.0012831381443047588</td>
        <td>16.0</td>
        <td>2.9955434374342885e-05</td>
        <td>4.0</td>
        <td>1.5693392617497486</td>
        <td>-</td>
    </tr>
    <tr>
        <td>ML1M</td>
        <td>betavae</td>
        <td>61</td>
        <td>50</td>
        <td>512</td>
        <td>0.0012652339274623833</td>
        <td>20.0</td>
        <td>6.585916758589679e-06</td>
        <td>4.0</td>
        <td>1.2616155423018485</td>
        <td>-</td>
    </tr>
    <tr>
        <td>ML1M</td>
        <td>betavae</td>
        <td>67</td>
        <td>500</td>
        <td>256</td>
        <td>0.0003284093433713744</td>
        <td>16.0</td>
        <td>0.00018661569142159942</td>
        <td>0.0</td>
        <td>1.117686056995137</td>
        <td>-</td>
    </tr>
    <tr>
        <td>ML1M</td>
        <td>betavae</td>
        <td>71</td>
        <td>500</td>
        <td>1024</td>
        <td>0.0008388350295324521</td>
        <td>13.0</td>
        <td>0.00013927091507350814</td>
        <td>0.0</td>
        <td>1.5102528680764704</td>
        <td>-</td>
    </tr>
    <tr>
        <td>ML1M</td>
        <td>macridvae</td>
        <td>41</td>
        <td>315</td>
        <td>1024</td>
        <td>7.997570944621344e-05</td>
        <td>8.0</td>
        <td>0.2811446327473092</td>
        <td>-</td>
        <td>3.95</td>
        <td>0.15000000000000002</td>
    </tr>
    <tr>
        <td>ML1M</td>
        <td>macridvae</td>
        <td>53</td>
        <td>180</td>
        <td>128</td>
        <td>0.00011408071636503805</td>
        <td>16.0</td>
        <td>0.015259295840668706</td>
        <td>-</td>
        <td>3.25</td>
        <td>0.125</td>
    </tr>
    <tr>
        <td>ML1M</td>
        <td>macridvae</td>
        <td>61</td>
        <td>90</td>
        <td>128</td>
        <td>0.0005151716645427616</td>
        <td>13.0</td>
        <td>3.0015312845898997e-05</td>
        <td>-</td>
        <td>3.25</td>
        <td>0.1</td>
    </tr>
    <tr>
        <td>ML1M</td>
        <td>macridvae</td>
        <td>67</td>
        <td>110</td>
        <td>128</td>
        <td>0.000336220968630433</td>
        <td>5.0</td>
        <td>0.0002392283318020879</td>
        <td>-</td>
        <td>2.0500000000000003</td>
        <td>0.1</td>
    </tr>
    <tr>
        <td>ML1M</td>
        <td>macridvae</td>
        <td>71</td>
        <td>185</td>
        <td>128</td>
        <td>0.00017442575061569994</td>
        <td>14.0</td>
        <td>1.8908338053953953e-05</td>
        <td>-</td>
        <td>0.5</td>
        <td>0.15000000000000002</td>
    </tr>
    <tr>
        <td>Yelp</td>
        <td>puresvd</td>
        <td>41</td>
        <td>0</td>
        <td>128</td>
        <td>0.1</td>
        <td>20.0</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>Yelp</td>
        <td>puresvd</td>
        <td>53</td>
        <td>0</td>
        <td>128</td>
        <td>0.1</td>
        <td>20.0</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>Yelp</td>
        <td>puresvd</td>
        <td>61</td>
        <td>0</td>
        <td>128</td>
        <td>0.1</td>
        <td>20.0</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>Yelp</td>
        <td>puresvd</td>
        <td>67</td>
        <td>0</td>
        <td>128</td>
        <td>0.1</td>
        <td>20.0</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>Yelp</td>
        <td>puresvd</td>
        <td>71</td>
        <td>0</td>
        <td>128</td>
        <td>0.1</td>
        <td>20.0</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>Yelp</td>
        <td>multidae</td>
        <td>41</td>
        <td>270</td>
        <td>1024</td>
        <td>0.0006938368881083159</td>
        <td>14.0</td>
        <td>0.00029463854548728203</td>
        <td>1.0</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>Yelp</td>
        <td>multidae</td>
        <td>53</td>
        <td>325</td>
        <td>1024</td>
        <td>0.0001805762620579476</td>
        <td>17.0</td>
        <td>0.0005395755347044184</td>
        <td>2.0</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>Yelp</td>
        <td>multidae</td>
        <td>61</td>
        <td>125</td>
        <td>128</td>
        <td>0.00034375847652142496</td>
        <td>18.0</td>
        <td>0.00021500643724408136</td>
        <td>1.0</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>Yelp</td>
        <td>multidae</td>
        <td>67</td>
        <td>170</td>
        <td>1024</td>
        <td>0.0007181407849740541</td>
        <td>13.0</td>
        <td>0.0002883424225524912</td>
        <td>2.0</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>Yelp</td>
        <td>multidae</td>
        <td>71</td>
        <td>280</td>
        <td>256</td>
        <td>0.0002609254474349184</td>
        <td>18.0</td>
        <td>0.00033649082628762534</td>
        <td>1.0</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>Yelp</td>
        <td>multivae</td>
        <td>41</td>
        <td>500</td>
        <td>1024</td>
        <td>0.0001445419728757848</td>
        <td>15.0</td>
        <td>0.0001529314021577996</td>
        <td>1.0</td>
        <td>0.85</td>
        <td>-</td>
    </tr>
    <tr>
        <td>Yelp</td>
        <td>multivae</td>
        <td>53</td>
        <td>175</td>
        <td>512</td>
        <td>0.0005076022119513504</td>
        <td>18.0</td>
        <td>0.0002153241852462664</td>
        <td>1.0</td>
        <td>0.08</td>
        <td>-</td>
    </tr>
    <tr>
        <td>Yelp</td>
        <td>multivae</td>
        <td>61</td>
        <td>100</td>
        <td>1024</td>
        <td>0.0004956230678115086</td>
        <td>17.0</td>
        <td>0.00013013403640565416</td>
        <td>2.0</td>
        <td>0.9</td>
        <td>-</td>
    </tr>
    <tr>
        <td>Yelp</td>
        <td>multivae</td>
        <td>67</td>
        <td>500</td>
        <td>512</td>
        <td>0.00012141635131171463</td>
        <td>11.0</td>
        <td>9.697402930782838e-05</td>
        <td>1.0</td>
        <td>0.33</td>
        <td>-</td>
    </tr>
    <tr>
        <td>Yelp</td>
        <td>multivae</td>
        <td>71</td>
        <td>205</td>
        <td>512</td>
        <td>0.002799283834705972</td>
        <td>20.0</td>
        <td>4.861539035090123e-05</td>
        <td>0.0</td>
        <td>0.17</td>
        <td>-</td>
    </tr>
    <tr>
        <td>Yelp</td>
        <td>betavae</td>
        <td>41</td>
        <td>215</td>
        <td>128</td>
        <td>0.00017401890043614913</td>
        <td>12.0</td>
        <td>1.2123125208621997e-05</td>
        <td>2.0</td>
        <td>1.9833360473363724</td>
        <td>-</td>
    </tr>
    <tr>
        <td>Yelp</td>
        <td>betavae</td>
        <td>53</td>
        <td>120</td>
        <td>512</td>
        <td>0.0017289484172260056</td>
        <td>17.0</td>
        <td>3.264509140903555e-05</td>
        <td>3.0</td>
        <td>1.7414053251436723</td>
        <td>-</td>
    </tr>
    <tr>
        <td>Yelp</td>
        <td>betavae</td>
        <td>61</td>
        <td>40</td>
        <td>512</td>
        <td>0.0009007034288638638</td>
        <td>20.0</td>
        <td>7.33914201054211e-06</td>
        <td>4.0</td>
        <td>1.1370287292320616</td>
        <td>-</td>
    </tr>
    <tr>
        <td>Yelp</td>
        <td>betavae</td>
        <td>67</td>
        <td>70</td>
        <td>512</td>
        <td>0.0012925541013861237</td>
        <td>11.0</td>
        <td>1.0774347459414433e-05</td>
        <td>4.0</td>
        <td>1.2527614930155757</td>
        <td>-</td>
    </tr>
    <tr>
        <td>Yelp</td>
        <td>betavae</td>
        <td>71</td>
        <td>500</td>
        <td>1024</td>
        <td>0.0004530447148995251</td>
        <td>19.0</td>
        <td>7.346284467654575e-06</td>
        <td>0.0</td>
        <td>1.266211766171639</td>
        <td>-</td>
    </tr>
    <tr>
        <td>Yelp</td>
        <td>macridvae</td>
        <td>41</td>
        <td>145</td>
        <td>512</td>
        <td>6.69522180659379e-05</td>
        <td>4.0</td>
        <td>0.0001658865359868799</td>
        <td>-</td>
        <td>0.25</td>
        <td>0.1</td>
    </tr>
    <tr>
        <td>Yelp</td>
        <td>macridvae</td>
        <td>53</td>
        <td>140</td>
        <td>256</td>
        <td>0.0001556427908050869</td>
        <td>12.0</td>
        <td>3.841414411957996e-05</td>
        <td>-</td>
        <td>3.1</td>
        <td>0.07500000000000001</td>
    </tr>
    <tr>
        <td>Yelp</td>
        <td>macridvae</td>
        <td>61</td>
        <td>80</td>
        <td>512</td>
        <td>7.08634551650617e-05</td>
        <td>14.0</td>
        <td>2.5322847712321084e-05</td>
        <td>-</td>
        <td>4.0</td>
        <td>0.07500000000000001</td>
    </tr>
    <tr>
        <td>Yelp</td>
        <td>macridvae</td>
        <td>67</td>
        <td>85</td>
        <td>512</td>
        <td>4.819376555214392e-05</td>
        <td>12.0</td>
        <td>0.0024185451263228287</td>
        <td>-</td>
        <td>2.1</td>
        <td>0.1</td>
    </tr>
    <tr>
        <td>Yelp</td>
        <td>macridvae</td>
        <td>71</td>
        <td>180</td>
        <td>512</td>
        <td>4.687921386948488e-05</td>
        <td>20.0</td>
        <td>0.00034575327047834233</td>
        <td>-</td>
        <td>1.4000000000000001</td>
        <td>0.07500000000000001</td>
    </tr>
    <tr>
        <td>GoodReadsChildren</td>
        <td>puresvd</td>
        <td>41</td>
        <td>0</td>
        <td>128</td>
        <td>0.1</td>
        <td>8.0</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>GoodReadsChildren</td>
        <td>puresvd</td>
        <td>53</td>
        <td>0</td>
        <td>128</td>
        <td>0.1</td>
        <td>8.0</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>GoodReadsChildren</td>
        <td>puresvd</td>
        <td>61</td>
        <td>0</td>
        <td>128</td>
        <td>0.1</td>
        <td>8.0</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>GoodReadsChildren</td>
        <td>puresvd</td>
        <td>67</td>
        <td>0</td>
        <td>128</td>
        <td>0.1</td>
        <td>8.0</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>GoodReadsChildren</td>
        <td>puresvd</td>
        <td>71</td>
        <td>0</td>
        <td>128</td>
        <td>0.1</td>
        <td>8.0</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>GoodReadsChildren</td>
        <td>multidae</td>
        <td>41</td>
        <td>85</td>
        <td>128</td>
        <td>0.0007380423704859757</td>
        <td>19.0</td>
        <td>0.00035590196011179876</td>
        <td>1.0</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>GoodReadsChildren</td>
        <td>multidae</td>
        <td>53</td>
        <td>165</td>
        <td>128</td>
        <td>0.0005958674388184799</td>
        <td>17.0</td>
        <td>0.0001639377341722471</td>
        <td>1.0</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>GoodReadsChildren</td>
        <td>multidae</td>
        <td>61</td>
        <td>425</td>
        <td>128</td>
        <td>4.988455707500659e-05</td>
        <td>18.0</td>
        <td>0.00046306422102069785</td>
        <td>2.0</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>GoodReadsChildren</td>
        <td>multidae</td>
        <td>67</td>
        <td>120</td>
        <td>512</td>
        <td>0.00022531480382998237</td>
        <td>18.0</td>
        <td>0.0002100343342643685</td>
        <td>3.0</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>GoodReadsChildren</td>
        <td>multidae</td>
        <td>71</td>
        <td>355</td>
        <td>256</td>
        <td>4.60692096421012e-05</td>
        <td>19.0</td>
        <td>0.0003250539863556743</td>
        <td>3.0</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>GoodReadsChildren</td>
        <td>multivae</td>
        <td>41</td>
        <td>115</td>
        <td>128</td>
        <td>0.00031488537486596373</td>
        <td>17.0</td>
        <td>4.634623131223256e-05</td>
        <td>1.0</td>
        <td>0.6</td>
        <td>-</td>
    </tr>
    <tr>
        <td>GoodReadsChildren</td>
        <td>multivae</td>
        <td>53</td>
        <td>165</td>
        <td>512</td>
        <td>0.0014330606547748113</td>
        <td>20.0</td>
        <td>0.00014525845778192055</td>
        <td>1.0</td>
        <td>0.21</td>
        <td>-</td>
    </tr>
    <tr>
        <td>GoodReadsChildren</td>
        <td>multivae</td>
        <td>61</td>
        <td>140</td>
        <td>256</td>
        <td>0.0003974152716813484</td>
        <td>18.0</td>
        <td>0.00039231648507585484</td>
        <td>1.0</td>
        <td>0.45</td>
        <td>-</td>
    </tr>
    <tr>
        <td>GoodReadsChildren</td>
        <td>multivae</td>
        <td>67</td>
        <td>100</td>
        <td>1024</td>
        <td>0.0014486482486958851</td>
        <td>19.0</td>
        <td>0.00011136269347878127</td>
        <td>1.0</td>
        <td>0.97</td>
        <td>-</td>
    </tr>
    <tr>
        <td>GoodReadsChildren</td>
        <td>multivae</td>
        <td>71</td>
        <td>160</td>
        <td>512</td>
        <td>0.00017905602461805515</td>
        <td>14.0</td>
        <td>0.000138045754842234</td>
        <td>2.0</td>
        <td>0.48</td>
        <td>-</td>
    </tr>
    <tr>
        <td>GoodReadsChildren</td>
        <td>betavae</td>
        <td>41</td>
        <td>95</td>
        <td>512</td>
        <td>0.0009290629433008043</td>
        <td>12.0</td>
        <td>7.197806785296871e-06</td>
        <td>2.0</td>
        <td>1.1486553800712236</td>
        <td>-</td>
    </tr>
    <tr>
        <td>GoodReadsChildren</td>
        <td>betavae</td>
        <td>53</td>
        <td>55</td>
        <td>512</td>
        <td>0.0010252967172160168</td>
        <td>17.0</td>
        <td>6.338260313234757e-06</td>
        <td>3.0</td>
        <td>1.2010337091181744</td>
        <td>-</td>
    </tr>
    <tr>
        <td>GoodReadsChildren</td>
        <td>betavae</td>
        <td>61</td>
        <td>500</td>
        <td>512</td>
        <td>9.700582078866563e-05</td>
        <td>12.0</td>
        <td>3.0224194047366592e-05</td>
        <td>1.0</td>
        <td>1.164924710819857</td>
        <td>-</td>
    </tr>
    <tr>
        <td>GoodReadsChildren</td>
        <td>betavae</td>
        <td>67</td>
        <td>50</td>
        <td>512</td>
        <td>0.0011886492634136402</td>
        <td>6.0</td>
        <td>3.086459414230785e-05</td>
        <td>3.0</td>
        <td>1.1421050831567254</td>
        <td>-</td>
    </tr>
    <tr>
        <td>GoodReadsChildren</td>
        <td>betavae</td>
        <td>71</td>
        <td>45</td>
        <td>128</td>
        <td>0.0006177349553171081</td>
        <td>10.0</td>
        <td>6.191554438156924e-06</td>
        <td>3.0</td>
        <td>1.223227291162441</td>
        <td>-</td>
    </tr>
    <tr>
        <td>GoodReadsChildren</td>
        <td>macridvae</td>
        <td>41</td>
        <td>55</td>
        <td>1024</td>
        <td>5.711877126048697e-05</td>
        <td>19.0</td>
        <td>0.16039646737694288</td>
        <td>-</td>
        <td>0.75</td>
        <td>0.05</td>
    </tr>
    <tr>
        <td>GoodReadsChildren</td>
        <td>macridvae</td>
        <td>53</td>
        <td>45</td>
        <td>256</td>
        <td>8.575239867359285e-05</td>
        <td>13.0</td>
        <td>2.219952091648614e-05</td>
        <td>-</td>
        <td>2.6500000000000004</td>
        <td>0.05</td>
    </tr>
    <tr>
        <td>GoodReadsChildren</td>
        <td>macridvae</td>
        <td>61</td>
        <td>70</td>
        <td>128</td>
        <td>4.9685806799172504e-05</td>
        <td>17.0</td>
        <td>0.00021939391981873875</td>
        <td>-</td>
        <td>1.25</td>
        <td>0.05</td>
    </tr>
    <tr>
        <td>GoodReadsChildren</td>
        <td>macridvae</td>
        <td>67</td>
        <td>80</td>
        <td>512</td>
        <td>5.936735284827353e-05</td>
        <td>13.0</td>
        <td>2.3077849513421525e-05</td>
        <td>-</td>
        <td>1.8</td>
        <td>0.05</td>
    </tr>
    <tr>
        <td>GoodReadsChildren</td>
        <td>macridvae</td>
        <td>71</td>
        <td>40</td>
        <td>256</td>
        <td>0.00030232412640953955</td>
        <td>18.0</td>
        <td>3.0466772416240343e-05</td>
        <td>-</td>
        <td>2.7</td>
        <td>0.07500000000000001</td>
    </tr>
</table>
