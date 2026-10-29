# Statistical Inference Lab: Viva Voce Preparation Guide

### Q1: What is the Central Limit Theorem and why is it foundational in inference?
**Answer:** The Central Limit Theorem states that as the sample size $n$ increases ($n \ge 30$), the sampling distribution of the sample mean $\bar{X}$ approaches a normal distribution, regardless of the shape of the underlying population distribution. This allows us to make parametric inferences (Z-tests, t-tests, confidence intervals) about the population mean $\mu$ even when the population data is skewed or non-normal.

### Q2: What is the difference between a Z-test and a t-test?
**Answer:**
- **Z-test:** Used when the population standard deviation $\sigma$ is known, or when the sample size is very large ($n > 30$).
- **t-test:** Used when the population standard deviation $\sigma$ is unknown and must be estimated using the sample standard deviation $s$. The t-distribution has heavier tails to account for extra sampling variance, with degrees of freedom $df = n - 1$.

### Q3: How do you interpret a p-value in hypothesis testing?
**Answer:** The p-value is the probability of obtaining test results at least as extreme as the observed results, assuming that the null hypothesis ($H_0$) is true. If $p < \alpha$ (where $\alpha = 0.05$ typically), we reject $H_0$ as the evidence is statistically significant. If $p \ge \alpha$, we fail to reject $H_0$.

### Q4: What is the purpose of a Chi-Square Test of Independence?
**Answer:** It assesses whether two categorical variables (e.g., Congestion Level and Weather Condition) are statistically independent. It compares observed cell frequencies $O_{ij}$ against expected cell frequencies $E_{ij} = (R_i \times C_j) / N$. If $\chi^2 = \sum \frac{(O - E)^2}{E}$ yields $p < 0.05$, the variables are dependent.

### Q5: When should you use Spearman's Rank Correlation instead of Pearson's?
**Answer:**
- **Pearson's $r$:** Measures the strength of a linear relationship between two continuous variables and assumes normality and no extreme outliers.
- **Spearman's $\rho$:** Measures monotonic relationships (whether variables increase together, linearly or non-linearly) using the ranks of values. It is non-parametric and robust against outliers and skewed distributions.

### Q6: What does Two-Way ANOVA tell you that One-Way ANOVA cannot?
**Answer:** Two-Way ANOVA evaluates the main effects of two independent categorical factors (e.g., Factor A: Weather Condition, Factor B: Peak Hour) simultaneously, as well as their **Interaction Effect** ($A \times B$), which tells us if the effect of one factor depends on the level of the other.

### Q7: What is the difference between $R^2$ and Adjusted $R^2$ in Regression?
**Answer:**
- $R^2$ represents the proportion of variance in the dependent variable explained by the independent variables. It always stays the same or increases when adding more features, even irrelevant ones.
- **Adjusted $R^2$** penalizes the model for adding predictors that do not improve goodness-of-fit, calculated as $1 - \frac{(1 - R^2)(n - 1)}{n - p - 1}$.
