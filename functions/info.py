def encoding_help():
    print("""
1. Type of analysis:

    ** No Encoding Needed:
        - Descriptive statistics, contingency tables, or non-parametric tests.
        - Decision Trees, Random Forests, Gradient Boosting (e.g., XGBoost) as these can handle categorical variables directly.
    
    ** Requires Encoding:
        - Linear Regression, Logistic Regression, Neural Networks, KNN, SVM, K-means.
        - Most distance-based algorithms require numeric data.

2. Type of Categorical Variable:

    ** Ordinal (ordered categories, e.g., low, medium, high):
        - Label Encoding is only necessary if the ordinal variables are not already numeric. 
          If they are already numeric and ordered, there is no need to encode them.

        - Use Label Encoding if the algorithm supports ordinal relationships (e.g., Linear/Logistic Regression).

        - Avoid One-Hot Encoding for ordinal variables, as it discards the ordinal information.

    ** Nominal (no inherent order, e.g., colors):
        - Apply One-Hot Encoding whether the categories are numeric or non-numeric to prevent the model from inferring an ordinal relationship where none exists.
        - One-Hot Encoding is generally preferred for nominal data, especially for algorithms like logistic regression or neural networks, as it converts categories into binary features.

3. Binary (Dichotomous) Variables:

    - If the variable has only two categories (e.g., 0/1, True/False):
        - No encoding is needed if the variable is already in binary form (e.g., 0 and 1). Most models will handle binary variables directly.
        - Avoid One-Hot Encoding for binary variables, as it introduces redundant features.

4. Advanced Techniques (optional):
    - For advanced applications, consider techniques like **Target Encoding** or **Frequency Encoding** for certain categorical variables in specific scenarios (e.g., high-cardinality nominal variables).
    """)


def Univariate_help():
    print("""
Pipeline Overview:

1. **Descriptive Analysis**:
   - **Continuous variables**: `analyse_descriptive_continuous` provides summary statistics like mean, standard deviation, and more for continuous data.
   - **Dichotomous variables**: `analyse_descriptive_dicho` counts frequencies and percentages for binary (dichotomous) categorical variables.

2. **Univariate Analysis**:
   - **Continuous variables**: Tests for normality (Shapiro), equality of variances (F-test), and comparison between two groups using T-tests, Mann-Whitney U, or Z-tests (depending on distribution and sample size).
   - **Categorical variables**: Chi-square or Fisher’s exact tests for contingency tables, calculation of Odds Ratio (OR) for 2x2 tables.
   - **Ordinal variables**: Kruskal-Wallis, Cochran-Armitage, or Mann-Whitney U tests for comparing ordinal data.

3. **Exporting Results**:
   - `export_descr` saves the descriptive statistics to an Excel file for easy access.

4. **ANOVA (Analysis of Variance)**:
   - If multiple groups need to be compared for continuous data, `perform_anova` conducts a one-way or factorial ANOVA to test differences across groups, including post-hoc Tukey tests for multiple comparisons.

5. **Significance Check**:
   - The `get_significant_variables` function identifies significant variables (p-value < 0.05) across continuous, categorical, and ordinal datasets from univariate tests.

This pipeline facilitates comprehensive univariate analysis by handling both descriptive and inferential statistics for various types of data (continuous, categorical, ordinal).
""")


def regression_interp_help(model_type="Logit"):
    print("""
            Result interpretation :
        1. Main results table 

        - **Coef. (Coefficient)**: Estimated regression coefficients for each independent variable (predictor). 
          They indicate the effect of each variable on the dependent variable. In linear regression, 
          a positive coefficient means that as the predictor increases, the dependent variable increases by the coefficient value.

        - **Std.Err. (Standard Error)**: This measures the variability of the coefficient estimate. 
          A large standard error indicates higher uncertainty in the estimate. Smaller standard errors indicate more precise estimates.

        - **t** (ols) / **z** (logit): The t-statistic for linear or z-statistic for logit is the ratio of the coefficient to its standard error.
          It is used to test if the coefficient is significantly different from zero. 
          Large values (in absolute terms) indicate that the coefficient is significantly different from zero.

        - **P>|t/z| (p-value)**: This is the probability that the observed statistic could have occurred under the null hypothesis (that the coefficient is 0). 
          If this value is small (e.g., less than 0.05), it means the coefficient is statistically significant and the variable likely has an impact on the dependent variable.

        - **[0.025 0.975] (95% Confidence Interval)**: This is the 95% confidence interval for the coefficient. 
          It shows the range within which the true coefficient is likely to fall with 95% confidence. 
          If the confidence interval contains 0, the coefficient is not statistically significant.""")
    if model_type == "Logit":
        print("""   
     2. Model summary table (Logit)
    
    - **Dependent Variable**: The dependent variable (i.e., the variable you are trying to predict).
    
    - **Model**: The type of model used (logit for logistic regression, ols for linear regression).
    
    - **Method**: The method used to estimate the coefficients (e.g., MLE - Maximum Likelihood Estimation).
    
    - **No. Observations**: The number of observations in the dataset (i.e., the number of rows or examples used in the analysis).
    Ensure that the number of observations in the final model does not fall below 70-80% of the original dataset size, 
    as this could indicate potential issues with missing data.
    If more than 20% of the data is missing for a given variable, it might be worth considering imputation or removal of that variable to maintain the model's integrity.
        
    - **Df Residuals**: The degrees of freedom of the residuals, calculated as the number of observations minus the number of estimated parameters.
            Usually close to the number of observations, 50-99% of observations.
    
    - **Df Model**: The number of estimated parameters in the model (i.e., the number of independent variables in the model plus the intercept (const)).
    
    - **Pseudo R-squared (logistic regression)**: Measures the goodness of fit of the model. There are several types of pseudo-R² 
      (such as McFadden's pseudo-R²); the closer this value is to 1, the better the model explains the data. However, pseudo-R² is often much lower than 
      the R² of linear regressions and should not be interpreted in the same way.
      Values between 0.02 and 0.30 are common; however, any value above 0.2 is generally considered indicative of a decent model fit in logistic regression.
    
    - **Log-Likelihood (LL) **: The value of the model’s log-likelihood. It measures the likelihood of observing the data given the estimated parameters. 
      The higher (less negative) the log-likelihood, the better the model fits.
      Ranges from -10 to -1000, depending on the dataset size and model.
      For a small dataset (e.g., fewer than 500 observations), log-likelihoods in the range of -100 to -500 are common, but lower values (closer to 0) are better.

    - **AIC (Akaike Information Criterion): AIC=2k−2LL with k parameters
          AIC is a measure used to evaluate the goodness of fit of a statistical model. 
          It balances how well the model fits the data and how complex the model is. AIC penalizes models with more parameters to avoid overfitting.
          In summary, AIC helps select a model that fits the data well but is not too complex. 
          The lower the AIC value, the better the model is, considering both fit and simplicity.

    - **BIC (Bayesian Information Criterion): BIC=ln(n)k−2LL with n observations
        BIC applies a stronger penalty for model complexity. 
        BIC is more conservative and penalizes models with more parameters more heavily than AIC. 
        This means BIC generally favors simpler models with fewer predictors unless there is strong evidence 
        that adding more complexity improves the model significantly. The lower the BIC value, the better the model is, 
        with an even greater emphasis on simplicity than AIC.
    
    Key Differences:
        Penalty for Complexity: BIC penalizes more for adding extra parameters than AIC. This makes BIC more conservative in selecting complex models.
        Application: AIC is often preferred when the focus is more on prediction accuracy, while BIC is used when model simplicity is more important.
        Interpretation: Both AIC and BIC aim to minimize values, but BIC tends to select simpler models compared to AIC due to its stricter penalty.

    3. Global statistical tests
    
    - **LLR (Likelihood Ratio) p-value**: This is a global test to see if the model as a whole is significant. 
         A small p-value means that the model as a whole is significantly better than the null model (which would have no independent variables).
        """)
    else:
        print("""
        2. Model summary table (OLS)

        - **Dependent Variable**: The dependent variable is the outcome you are trying to predict.

        - **Model**: The type of model used (OLS for ordinary least squares regression).

        - **Method**: The method used to estimate the coefficients (OLS for linear regression).

        - **No. Observations**: The number of observations in the dataset (i.e., the number of rows or examples used in the analysis).
        Ensure that the number of observations in the final model does not fall below 70-80% of the original dataset size, 
        as this could indicate potential issues with missing data.
        If more than 20% of the data is missing for a given variable, it might be worth considering imputation or removal of that variable to maintain the model's integrity.
            
        - **Df Residuals**: The degrees of freedom of the residuals, calculated as the number of observations minus the number of estimated parameters.
          Usually close to the number of observations, 50-99% of observations.

        - **Df Model**: The number of estimated parameters in the model (i.e., the number of independent variables in the model plus the intercept (const)).

        - **R-squared**: R-squared measures the proportion of the variance in the dependent variable that is explained by the independent variables. 
          Values range from 0 to 1, with higher values indicating a better fit. For example, an R-squared of 0.7 means that 70% of the variation in the dependent variable is explained by the model. 

        - **Adj. R-squared**: Adjusted R-squared is similar to R-squared but adjusts for the number of predictors in the model. 
          It accounts for the fact that adding more variables to the model will always increase R-squared, even if the variables don't improve the model. 
          Use adjusted R-squared to compare models with different numbers of predictors.

        - **F-statistic**: The F-statistic tests the overall significance of the model. 
          It tells us whether at least one of the independent variables in the model is statistically significant. A large F-statistic (and a small p-value) indicates that the model as a whole is significant.

        - **Prob (F-statistic)**: The p-value associated with the F-statistic. If this value is small (e.g., less than 0.05), it indicates that the model as a whole is statistically significant and better than a model with no predictors.

        - **Log-Likelihood (LL) **: The value of the model’s log-likelihood. It measures the likelihood of observing the data given the estimated parameters. 
          The higher (less negative) the log-likelihood, the better the model fits.
          Ranges from -10 to -1000, depending on the dataset size and model.
          For a small dataset (e.g., fewer than 500 observations), log-likelihoods in the range of -100 to -500 are common, but lower values (closer to 0) are better.
          
        - **AIC (Akaike Information Criterion): AIC=2k−2LL with k parameters
            AIC is a measure used to evaluate the goodness of fit of a statistical model. 
            It balances how well the model fits the data and how complex the model is. AIC penalizes models with more parameters to avoid overfitting.
            In summary, AIC helps select a model that fits the data well but is not too complex. 
            The lower the AIC value, the better the model is, considering both fit and simplicity.
        
        - **BIC (Bayesian Information Criterion): BIC=ln(n)k−2LL with n observations
            BIC applies a stronger penalty for model complexity. 
            BIC is more conservative and penalizes models with more parameters more heavily than AIC. 
            This means BIC generally favors simpler models with fewer predictors unless there is strong evidence 
            that adding more complexity improves the model significantly. The lower the BIC value, the better the model is, 
            with an even greater emphasis on simplicity than AIC.
        
        Key Differences:
            Penalty for Complexity: BIC penalizes more for adding extra parameters than AIC. This makes BIC more conservative in selecting complex models.
            Application: AIC is often preferred when the focus is more on prediction accuracy, while BIC is used when model simplicity is more important.
            Interpretation: Both AIC and BIC aim to minimize values, but BIC tends to select simpler models compared to AIC due to its stricter penalty.

        3. Global statistical tests

        - **F-statistic (and Prob > F)**: This is a global test to see if the model as a whole is statistically significant. 
          It tests whether at least one predictor is significant. A small p-value associated with the F-statistic means that the model as a whole is better than one without any predictors (null model).

        - **Durbin-Watson (D-W)**: The Durbin-Watson statistic tests for the presence of autocorrelation in the residuals. 
          Values close to 2 indicate no autocorrelation, while values closer to 0 or 4 suggest positive or negative autocorrelation, respectively.

        - **Cond. No (Condition Number)**: This tests for multicollinearity, which occurs when independent variables are highly correlated. 
          A large condition number (e.g., above 30) may indicate that there is strong multicollinearity in the model, which could make the estimates unstable.
            """)

def random_effect_help():
    """
       Helper function to explain how to parameterize a hierarchical linear model (HLM),
       focusing on when and why to use `re_formula` and `vc_formula` for random effects.

       Returns:
       - A detailed explanation of the use of `re_formula` and `vc_formula` in HLMs.
       """
    help_text = """
       ### When to Use re_formula:
       - **What it does**: `re_formula` allows you to specify **random effects**, which means you can model how different groups in your data (like schools, hospitals, companies) have variations that affect the outcome (random intercepts or slopes).

       - **When to use it**:
         - If you want to **add random intercepts** for specific groups (e.g., each school has a different baseline performance), use `re_formula`.
         - If you want to **add random slopes** (e.g., the effect of study hours varies across different schools), you can include variables that should have varying slopes in `re_formula`.

       - **Examples**:
         - **Random intercept for each school**: If students are nested within schools, you might want each school to have a different intercept. Set `re_formula = "0 + C(school_id)"` to suppress the global intercept and only model school-specific intercepts.
         - **Random intercept and random slope**: If the effect of `hours_studied` on test scores varies by school, you could use `re_formula = "0 + C(school_id) + hours_studied:C(school_id)"`, meaning each school has a different baseline (intercept) and a different slope for `hours_studied`.

       - **When NOT to use it**:
         - If you’re okay with just a **random intercept** for the group specified by `groups` in the model (e.g., each school has a different intercept but no random slope), you don’t need to specify `re_formula`. It defaults to random intercepts based on the `groups`.

       ### When to Use vc_formula:
       - **What it does**: `vc_formula` (variance components) allows you to model more complex structures where there are **multiple sources of variability** or random effects at multiple levels (e.g., hospitals within cities, patients within hospitals).

       - **When to use it**:
         - If you want to model **nested structures** (e.g., students nested within classrooms nested within schools), use `vc_formula`. You can specify different sources of variance for each level.
         - If you need **multiple random effects** for different levels of your data hierarchy (e.g., schools, classes, and teachers all contribute different random effects).

       - **Examples**:
         - **Variance for multiple groups**: Let’s say you want to model random intercepts for both `school_id` and `teacher_id`. You could use `vc_formula = {"school_id": "0 + C(school_id)", "teacher_id": "C(teacher_id)"}`. This means that both schools and teachers contribute different sources of variability.

       - **When NOT to use it**:
         - If you only need simple random intercepts for the grouping variable (e.g., only `school_id`), `vc_formula` might be unnecessary. Stick to `re_formula` or even just the default behavior of random intercepts.

       ### Quick Summary:
       - Use **`re_formula`** to specify **random effects** (e.g., random intercepts or slopes) for specific groups or variables.
       - Use **`vc_formula`** if you have a more complex **nested structure** and need to model multiple sources of variability.

       ### Example Scenarios:
       - **Simple Random Intercept**: Students are nested in schools, and each school has a different baseline performance. You don’t need to specify `re_formula` - just pass `groups="school_id"`.

       - **Random Intercept and Slope**: The effect of study hours varies across schools. Use `re_formula = "0 + C(school_id) + hours_studied:C(school_id)"` to model both random intercepts and random slopes for study hours across schools.

       - **Multiple Random Effects**: Students nested in schools, and classes within schools also contribute variability. Use `vc_formula = {"school_id": "0 + C(school_id)", "class_id": "C(class_id)"}` to account for both school- and class-level random effects.
       """
    print(help_text)


def hlm_help(random_effects_infos=True):
    """
    Helper function to explain the general use of hierarchical linear models (HLMs),
    including key components like formula, groups, re_formula, vc_formula,
    and when to use HLMs.

    Returns:
    - A detailed explanation of HLM usage and parameters.
    """
    help_text = """
    ### What are Hierarchical Linear Models (HLMs)?
    Hierarchical Linear Models (HLMs), also known as mixed models or multilevel models, 
    are used when you have data that is organized at more than one level (e.g., 
    students within classrooms, patients within hospitals). HLMs account for both 
    fixed effects (the main variables of interest) and random effects (group-specific 
    variations).

    ### When to Use HLMs:
    - When your data has a **nested or hierarchical structure** (e.g., students nested within schools, 
      patients nested within clinics).
    - When you suspect there are **group-level effects** that should be treated as random (e.g., 
      schools may have different baseline performances, but you are not specifically interested 
      in each individual school’s effect).
    - When you need to model **both fixed effects** (like hours studied) and **random effects** 
      (like variability between schools).

    ### Key Components of HLM:
    1. **formula**:
       - The main formula specifying the relationship between the dependent variable and the fixed effects.
       - Example: `"outcome ~ predictor1 + predictor2"`

    2. **groups**:
       - This defines the grouping structure (e.g., `school_id`, `clinic_id`).
       - It tells the model at what level the random effects should be applied.
       - Example: `groups="school_id"` will treat schools as random effects.

    3. **re_formula** (Random Effects Formula):
       - This defines which variables should have **random effects**.
       - If left unspecified, a **random intercept** is assumed for the grouping variable provided in `groups`.
       - Use `re_formula` if you want more control over the random effects (e.g., random slopes, random intercepts).
       - Example:
         - **Random intercept only**: `"0 + C(school_id)"` means there is a different intercept for each school.
         - **Random intercept and slope**: `"0 + C(school_id) + predictor1:C(school_id)"` means both intercepts 
           and slopes for `predictor1` can vary by school.

    4. **vc_formula** (Variance Components Formula):
       - This is for more **complex random effect structures** where you have **multiple sources of variability** 
         or hierarchical random effects.
       - Use `vc_formula` to specify multiple random effects (e.g., for both school and classroom levels).
       - Example: `{"school_id": "0 + C(school_id)", "classroom_id": "C(classroom_id)"}` would model 
         both school- and classroom-level random effects.

    ### Example HLM Setup:
    Suppose you have a dataset where students are nested within schools, and you want to model test scores 
    (`test_score`) based on the number of hours studied (`hours_studied`), while accounting for the fact 
    that both schools and classrooms may have random effects.

    - **formula**: `"test_score ~ hours_studied"`
    - **groups**: `"school_id"` to model random intercepts for each school.
    - **re_formula**: `"0 + C(school_id) + hours_studied:C(school_id)"` to model random intercepts and random 
      slopes for study hours across schools.
    - **vc_formula**: `{"school_id": "0 + C(school_id)", "classroom_id": "C(classroom_id)"}` to account for 
      random effects at both the school and classroom levels.

    ### Key Points:
    - **Use HLMs when you need to model data with hierarchical structures**, such as students nested within schools.
    - **Specify `groups` for random intercepts** at the group level.
    - **Use `re_formula` if you need to add random slopes** or more complex random effects.
    - **Use `vc_formula` for multiple sources of variability** or nested random effects.

    """
    print(help_text)

    if random_effects_infos:
        random_effect_help()
