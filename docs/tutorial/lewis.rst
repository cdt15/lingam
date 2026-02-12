Causal Explanations with LEWIS
==============================

This notebook implements **LEWIS**, a causal explanation framework for
black-box machine learning models. Using probabilistic contrastive
counterfactuals, we compute **Necessity (Nec)**, **Sufficiency (Suf)**,
and **Necessity-and-Sufficiency (NeSuf)** scores to quantify causal
feature importance without relying on model internals. This notebook
follows the methodology proposed by Galhotra et al. (SIGMOD 2021).

**Reference:** 

* Sainyam Galhotra, Romila Pradhan, Babak Salimi (2021). Explaining Black-Box Algorithms Using Probabilistic Contrastive Counterfactuals. SIGMOD ’21: International Conference on Management of Data, Virtual Event, China, June 20-25, 2021.

.. code-block:: python

    import numpy as np
    import pandas as pd
    import operator
    import graphviz
    import matplotlib.pyplot as plt
    
    from sklearn.preprocessing import KBinsDiscretizer
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    import lingam
    from lingam.utils import make_dot
    
    print([np.__version__, pd.__version__, graphviz.__version__, lingam.__version__])
    
    np.set_printoptions(precision=3, suppress=True)
    np.random.seed(0)


.. parsed-literal::

    ['1.26.4', '2.3.0', '0.20.3', '1.12.1']
    

Test data
---------

We create test data consisting of 7 variables.

.. code-block:: python

    m = np.array([
        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [ 0.4, 0.0,-0.6, 0.0, 0.0, 0.0, 0.0],
        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0],
        [ 0.0, 0.2, 0.0, 0.0, 0.0,-0.7, 0.0],
        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [ 0.0, 0.5, 0.0,-0.1,-0.5, 0.0, 0.0],
    ])
    
    generate_error = lambda p: np.random.uniform(-p, p, size=1000)
    
    error_vars = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    params = [0.5 * np.sqrt(12 * v) for v in error_vars]
    e = np.array([generate_error(p) for p in params])
    
    cols = [f"x{i}" for i in range(len(m) - 1)] + ["y"]
    X = np.linalg.pinv(np.eye(len(m)) - m) @ e
    df = pd.DataFrame(X.T, columns=cols)
    
    display(make_dot(m, labels=cols))
    df.head()



.. image:: ../image/lewis.svg




.. raw:: html

    <div>
    <style scoped>
        .dataframe {
            font-family: verdana, arial, sans-serif;
            font-size: 11px;
            color: #333333;
            border-width: 1px;
            border-color: #B3B3B3;
            border-collapse: collapse;
        }
        .dataframe thead th {
            border-width: 1px;
            padding: 8px;
            border-style: solid;
            border-color: #B3B3B3;
            background-color: #B3B3B3;
        }
        .dataframe tbody th {
            border-width: 1px;
            padding: 8px;
            border-style: solid;
            border-color: #B3B3B3;
        }
        .dataframe tr:nth-child(even) th{
        background-color: #EAEAEA;
        }
        .dataframe tr:nth-child(even) td{
            background-color: #EAEAEA;
        }
        .dataframe td {
            border-width: 1px;
            padding: 8px;
            border-style: solid;
            border-color: #B3B3B3;
            background-color: #ffffff;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>x0</th>
          <th>x1</th>
          <th>x2</th>
          <th>x3</th>
          <th>x4</th>
          <th>x5</th>
          <th>y</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0.119568</td>
          <td>-0.182500</td>
          <td>0.763061</td>
          <td>-0.373908</td>
          <td>-0.315997</td>
          <td>-0.326320</td>
          <td>-0.202553</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.527104</td>
          <td>-0.954103</td>
          <td>-0.058582</td>
          <td>0.069640</td>
          <td>0.319117</td>
          <td>-0.495717</td>
          <td>-0.818257</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.251718</td>
          <td>0.007441</td>
          <td>0.056720</td>
          <td>0.154034</td>
          <td>-0.146964</td>
          <td>-1.056711</td>
          <td>0.510195</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.109941</td>
          <td>0.922016</td>
          <td>-0.611097</td>
          <td>0.680521</td>
          <td>0.069137</td>
          <td>-0.361232</td>
          <td>0.605658</td>
        </tr>
        <tr>
          <th>4</th>
          <td>-0.187007</td>
          <td>-1.346211</td>
          <td>0.257302</td>
          <td>0.447058</td>
          <td>-0.904446</td>
          <td>-0.655983</td>
          <td>-1.317686</td>
        </tr>
      </tbody>
    </table>
    </div>
    <br>


Discretization
--------------

LEWIS assumes that all features have finite, discrete domains.
Therefore, continuous-valued features must be discretized before
computing LEWIS scores. Each continuous feature is binned into a small
number of ordered categories (e.g., Low < Medium < High), enabling
contrastive value pairs :math:`(x, x′)` and well-defined computation of
Necessity, Sufficiency, and Necessity-and-Sufficiency scores.

.. code-block:: python

    kbd = KBinsDiscretizer(n_bins=2, encode="ordinal", strategy="uniform", subsample=None)
    df["y"] = kbd.fit_transform(df["y"].values.reshape(-1, 1))
    
    feature_names = [f"x{i}" for i in range(len(m) - 1)]
    for name in feature_names:
        kbd = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="uniform", subsample=None)
        df[name] = kbd.fit_transform(df[name].values.reshape(-1, 1))
    
    df.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe {
            font-family: verdana, arial, sans-serif;
            font-size: 11px;
            color: #333333;
            border-width: 1px;
            border-color: #B3B3B3;
            border-collapse: collapse;
        }
        .dataframe thead th {
            border-width: 1px;
            padding: 8px;
            border-style: solid;
            border-color: #B3B3B3;
            background-color: #B3B3B3;
        }
        .dataframe tbody th {
            border-width: 1px;
            padding: 8px;
            border-style: solid;
            border-color: #B3B3B3;
        }
        .dataframe tr:nth-child(even) th{
        background-color: #EAEAEA;
        }
        .dataframe tr:nth-child(even) td{
            background-color: #EAEAEA;
        }
        .dataframe td {
            border-width: 1px;
            padding: 8px;
            border-style: solid;
            border-color: #B3B3B3;
            background-color: #ffffff;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>x0</th>
          <th>x1</th>
          <th>x2</th>
          <th>x3</th>
          <th>x4</th>
          <th>x5</th>
          <th>y</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>2.0</td>
          <td>2.0</td>
          <td>4.0</td>
          <td>1.0</td>
          <td>2.0</td>
          <td>1.0</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>1</th>
          <td>3.0</td>
          <td>1.0</td>
          <td>2.0</td>
          <td>2.0</td>
          <td>2.0</td>
          <td>1.0</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>2</th>
          <td>3.0</td>
          <td>2.0</td>
          <td>2.0</td>
          <td>2.0</td>
          <td>2.0</td>
          <td>0.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>3</th>
          <td>2.0</td>
          <td>3.0</td>
          <td>1.0</td>
          <td>3.0</td>
          <td>2.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>4</th>
          <td>2.0</td>
          <td>1.0</td>
          <td>3.0</td>
          <td>3.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>0.0</td>
        </tr>
      </tbody>
    </table>
    </div>
    <br>


Model Training and Prediction with RandomForestClassifier
---------------------------------------------------------

We train a black-box prediction model using a RandomForestClassifier.
The model learns a non-linear decision function by aggregating
predictions from multiple decision trees trained on bootstrapped samples
of the data. After training, the model is used to generate predictions
(and class probabilities) for each instance. These predicted outcomes
serve as the decision variable used in subsequent LEWIS analysis, while
the internal structure of the model remains opaque.

.. code-block:: python

    X = df[feature_names]
    y = df["y"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
    
    clf = RandomForestClassifier(max_depth=5, random_state=0)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    X_test["y"] = y_pred
    X_test.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe {
            font-family: verdana, arial, sans-serif;
            font-size: 11px;
            color: #333333;
            border-width: 1px;
            border-color: #B3B3B3;
            border-collapse: collapse;
        }
        .dataframe thead th {
            border-width: 1px;
            padding: 8px;
            border-style: solid;
            border-color: #B3B3B3;
            background-color: #B3B3B3;
        }
        .dataframe tbody th {
            border-width: 1px;
            padding: 8px;
            border-style: solid;
            border-color: #B3B3B3;
        }
        .dataframe tr:nth-child(even) th{
        background-color: #EAEAEA;
        }
        .dataframe tr:nth-child(even) td{
            background-color: #EAEAEA;
        }
        .dataframe td {
            border-width: 1px;
            padding: 8px;
            border-style: solid;
            border-color: #B3B3B3;
            background-color: #ffffff;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>x0</th>
          <th>x1</th>
          <th>x2</th>
          <th>x3</th>
          <th>x4</th>
          <th>x5</th>
          <th>y</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>993</th>
          <td>2.0</td>
          <td>2.0</td>
          <td>4.0</td>
          <td>2.0</td>
          <td>1.0</td>
          <td>2.0</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>859</th>
          <td>1.0</td>
          <td>0.0</td>
          <td>4.0</td>
          <td>3.0</td>
          <td>2.0</td>
          <td>1.0</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>298</th>
          <td>4.0</td>
          <td>3.0</td>
          <td>4.0</td>
          <td>1.0</td>
          <td>4.0</td>
          <td>1.0</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>553</th>
          <td>2.0</td>
          <td>1.0</td>
          <td>4.0</td>
          <td>2.0</td>
          <td>0.0</td>
          <td>4.0</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>672</th>
          <td>3.0</td>
          <td>2.0</td>
          <td>2.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>2.0</td>
          <td>1.0</td>
        </tr>
      </tbody>
    </table>
    </div>
    <br>


Computing LEWIS Scores
----------------------

We compute LEWIS explanation scores—**Necessity (Nec)**, **Sufficiency
(Suf)**, and **Necessity-and-Sufficiency (NeSuf)**—for each selected
feature using a causal, model-agnostic approach. Based on the given
causal graph, we first identify an adjustment set that satisfies the
**backdoor criterion** for the relationship between the target feature
and the outcome. These variables are used only for probability
estimation and are not part of the user-facing explanation.

For an ordinal feature with ordered values
:math:`(x^{(1)} \prec \dots \prec x^{(L)})`, we compute Nec, Suf, and
NeSuf for **all ordered value pairs** :math:`(x, x')` such that
:math:`(x \succ x')`. After evaluating all pairs, we select the pair
that **maximizes NeSuf** as the most causally responsive contrast. The
corresponding Nec and Suf values for this same pair are then reported,
ensuring that all three scores are aligned to a single, interpretable
intervention.

.. code-block:: python

    backdoor = {
        'x0': [],
        'x1': [],
        'x2': [],
        'x3': ['x5'],
        'x4': ['x1', 'x5'],
        'x5': [],
    }

.. code-block:: python

    def max_nesuf(score_dic, score, name):
        pre_score = [0, 0, 0]
        if name in score_dic.keys():
            pre_score = score_dic[name]
        if score[2] > pre_score[2]: # nesuf
            score_dic[name] = score
        return score_dic

.. code-block:: python

    score_dic = {}
    for name in feature_names:
        val = {}
        for v in df[name].unique():
            df_temp = df[df[name] == v]
            vc = df_temp["y"].value_counts().reindex([0, 1], fill_value=0)
            val[v] = vc[1] * 1.0 / (vc[0] + vc[1])
        sorted_val = sorted(val.items(), key=operator.itemgetter(1))
    
        for i, (before, _) in enumerate(sorted_val):
            for after, _ in sorted_val[i + 1 :]:
                lewis = lingam.LEWIS()
                score = lewis.get_scores(
                    X_test,
                    x_names=[name],
                    x_values=[after],
                    x_prime_values=[before],
                    o_name="y", 
                    c_names=backdoor[name])
                score_dic = max_nesuf(score_dic, score, name)
                print(f"{name}: {before}->{after}: score={score}")
    


.. parsed-literal::

    x0: 0.0->1.0: score=(0.28876449588651437, 0.15671962820931234, 0.11307290998254543)
    x0: 0.0->2.0: score=(0.37054073719681124, 0.22722774483181898, 0.16394438035923292)
    x0: 0.0->3.0: score=(0.308951540247843, 0.1725738099335212, 0.12451167156864434)
    x0: 0.0->4.0: score=(0.3041121895598342, 0.16868933442058193, 0.12170902996578553)
    x0: 1.0->2.0: score=(0.11497772655799332, 0.08361171323457248, 0.05087147037668749)
    x0: 1.0->3.0: score=(0.028383066149784867, 0.018800605652118815, 0.011438761586098911)
    x0: 1.0->4.0: score=(0.02157891947822522, 0.014194218923715932, 0.008636119983240098)
    x0: 2.0->3.0: score=(0.0, 0.0, 0.0)
    x0: 2.0->4.0: score=(0.0, 0.0, 0.0)
    x0: 3.0->4.0: score=(0.0, 0.0, 0.0)
    x1: 0.0->1.0: score=(1.0, 0.058517831098751374, 0.058517831098751374)
    x1: 0.0->2.0: score=(1.0, 0.3863071221311797, 0.3863071221311797)
    x1: 0.0->3.0: score=(1.0, 0.6916588068976024, 0.6916588068976024)
    x1: 0.0->4.0: score=(1.0, 0.610516293905489, 0.610516293905489)
    x1: 1.0->2.0: score=(0.8485199269018905, 0.348163036815634, 0.3277892910324283)
    x1: 1.0->3.0: score=(0.9153949454338188, 0.6724938577835782, 0.633140975798851)
    x1: 1.0->4.0: score=(0.9041502549843325, 0.5863079313025591, 0.5519984628067376)
    x1: 2.0->3.0: score=(0.44147733206212025, 0.4975643286375128, 0.3053516847664227)
    x1: 2.0->4.0: score=(0.36724518905144576, 0.3653442623497988, 0.22420917177430932)
    x1: 3.0->4.0: score=(0.0, 0.0, 0.0)
    x2: 4.0->3.0: score=(0.37781207464998834, 0.12785476560870568, 0.10561677393784988)
    x2: 4.0->2.0: score=(0.5960755908256189, 0.31071620571720965, 0.2566728201473194)
    x2: 4.0->0.0: score=(0.6241556609044814, 0.3496612983303619, 0.2888441281382971)
    x2: 4.0->1.0: score=(0.6962790808358484, 0.48269334060441804, 0.39873768641480434)
    x2: 3.0->2.0: score=(0.350799987082434, 0.2096685653922427, 0.1510560462094695)
    x2: 3.0->0.0: score=(0.3959311587667224, 0.2543229315200742, 0.18322735420044717)
    x2: 3.0->1.0: score=(0.5118501873958845, 0.406857207955013, 0.2931209124769545)
    x2: 2.0->0.0: score=(0.06951813121731897, 0.056500809878571416, 0.03217130799097767)
    x2: 2.0->1.0: score=(0.2480748569145521, 0.2495012015568321, 0.142064866267485)
    x2: 0.0->1.0: score=(0.19189704999930096, 0.2045580893963687, 0.10989355827650732)
    x3: 0.0->3.0: score=(0.0, 0.0, 0.07576064369254501)
    x3: 0.0->2.0: score=(0.24019885850208111, 0.0, 0.06756468929060586)
    x3: 0.0->4.0: score=(0.4375471879801787, 0.33627411063087126, 0.2394474103512867)
    x3: 0.0->1.0: score=(0.5215173421020096, 0.2077705378411908, 0.2560619882491983)
    x3: 3.0->2.0: score=(5.8380840876851114e-05, 0.0, 0.0016793883085030473)
    x3: 3.0->4.0: score=(0.06087756395956861, 0.18097315391122015, 0.1735621093691839)
    x3: 3.0->1.0: score=(0.4050632226945426, 0.2552560187517776, 0.1901766872670955)
    x3: 2.0->4.0: score=(0.08389050626406856, 0.26072855163571107, 0.17188272106068087)
    x3: 2.0->1.0: score=(0.39325969742974465, 0.27799227073061517, 0.18849729895859244)
    x3: 4.0->1.0: score=(0.009866759685432035, 0.3007171788408852, 0.016614577897911605)
    x4: 4.0->3.0: score=(0.0, 0.0002437752006358201, 0.01417600649350649)
    x4: 4.0->2.0: score=(0.0, 0.43864694304558277, 0.3479847269469221)
    x4: 4.0->1.0: score=(0.36470213549225855, 0.6477582075236524, 0.5560876613241783)
    x4: 4.0->0.0: score=(0.8540240160406432, 0.6193767844978033, 0.624207928623127)
    x4: 3.0->2.0: score=(0.0, 0.3662085477202382, 0.33380872045341564)
    x4: 3.0->1.0: score=(0.34372689656444605, 0.6086744374259367, 0.5419116548306718)
    x4: 3.0->0.0: score=(0.8449920368334224, 0.6353532663565686, 0.6100319221296207)
    x4: 2.0->1.0: score=(0.0, 0.408105331786863, 0.21286296955041636)
    x4: 2.0->0.0: score=(0.24400727130132333, 0.5372555633357295, 0.27622320167620484)
    x4: 1.0->0.0: score=(0.17773403145390043, 0.252226191821934, 0.07080119345612482)
    x5: 1.0->0.0: score=(0.0, 0.0, 0.0)
    x5: 1.0->2.0: score=(0.2233835942176238, 0.08602778414260957, 0.06622184994605693)
    x5: 1.0->3.0: score=(0.5867549758794374, 0.42466177557054957, 0.3268930922716893)
    x5: 1.0->4.0: score=(0.6595932839891901, 0.5795249365522417, 0.44610254432152086)
    x5: 0.0->2.0: score=(0.4502876323702336, 0.1594757752886144, 0.13348733208373778)
    x5: 0.0->3.0: score=(0.7074927867488406, 0.4708967005285375, 0.3941585744093701)
    x5: 0.0->4.0: score=(0.7590499242327206, 0.613314857297541, 0.5133680264592017)
    x5: 2.0->3.0: score=(0.46789042692929883, 0.37050797119720974, 0.2606712423256323)
    x5: 2.0->4.0: score=(0.5616797256969114, 0.5399476525078897, 0.3798806943754639)
    x5: 3.0->4.0: score=(0.17625937121629423, 0.2691689069247335, 0.11920945204983158)
    

Visualization of LEWIS Scores Sorted by NeSuf
---------------------------------------------

Finally, we visualize the LEWIS scores using a horizontal bar chart,
where features are **sorted in descending order of NeSuf**. NeSuf
represents the strongest indicator of causal importance, capturing how
often a feature change would flip the decision outcome in both
directions. By sorting features by NeSuf, the plot highlights which
features are most causally influential overall. For each feature, the
corresponding **Nec and Suf values for the same maximizing contrast**
are displayed alongside NeSuf, enabling a consistent and interpretable
comparison.

.. code-block:: python

    def plot_scores_barh_featurecol(
        df,
        feature_col="feature",
        value_cols=("Nec", "Suf", "NeSuf"),
        title="LEWIS Scores",
        reverse_for_image=True,
    ):
        needed = [feature_col, *value_cols]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"df is missing columns {missing}. Required columns: {needed}")
    
        df_plot = (
            df.loc[:, needed]
            .dropna(subset=[feature_col])
            .drop_duplicates(subset=[feature_col])
            .set_index(feature_col)
            .loc[:, list(value_cols)]
        )
    
        if reverse_for_image:
            df_plot = df_plot.iloc[::-1]
    
        y = np.arange(len(df_plot.index))
        h = 0.22 
    
        fig, ax = plt.subplots(figsize=(3.0, 3.6), dpi=120)
    
        bars1 = ax.barh(
            y + h,
            df_plot[value_cols[0]].values,
            height=h,
            color="#4C72B0",
            edgecolor="black",
            linewidth=0.6,
            label=value_cols[0],
        )
        bars2 = ax.barh(
            y,
            df_plot[value_cols[1]].values,
            height=h,
            color="#F0E442",
            edgecolor="black",
            linewidth=0.6,
            label=value_cols[1],
        )
        bars3 = ax.barh(
            y - h,
            df_plot[value_cols[2]].values,
            height=h,
            color="#2CA02C",
            edgecolor="black",
            linewidth=0.6,
            label=value_cols[2],
        )
    
        ax.set_yticks(y)
        ax.set_yticklabels(df_plot.index)
        ax.set_xlim(0, 1.0)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.set_xlabel(title)
        ax.legend(loc="lower right", fontsize=7, frameon=True)
    
        def add_labels(bars):
            for b in bars:
                w = b.get_width()
                yy = b.get_y() + b.get_height() / 2
                ax.text(w + 0.01, yy, f"{w:.2f}", va="center", ha="left", fontsize=7)
    
        add_labels(bars1)
        add_labels(bars2)
        add_labels(bars3)
    
        ax.grid(False)
        plt.tight_layout()
        plt.show()

.. code-block:: python

    nec = []
    suf = []
    nesuf = []
    for name in feature_names:
        score = score_dic[name]
        nec.append(score[0])
        suf.append(score[1])
        nesuf.append(score[2])
    score_df = pd.DataFrame()
    score_df['feature'] = feature_names
    score_df['Nec'] = nec
    score_df['Suf'] = suf
    score_df['NeSuf'] = nesuf
    
    plot_scores_barh_featurecol(score_df.sort_values(['NeSuf'], ascending=False))



.. image:: ../image/lewis.png


