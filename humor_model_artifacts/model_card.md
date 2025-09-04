
# HUMOR DETECTION MODEL CARD
*Generated: 2025-08-25 18:35:46*

## MODEL OVERVIEW

### Architecture
- **Model Type**: Multinomial Logistic Regression with Platt Scaling Calibration
- **Features**: Text (TF-IDF + linguistic) + Demographics (age, gender, ethnicity)
- **Target**: 3-class humor classification (funny, not_funny, dont_understand)
- **Framework**: scikit-learn with CalibratedClassifierCV

### Performance Summary
- **Primary Metric**: F1-Macro (handles class imbalance)
- **Best Performance**: 0.544 F1-macro
- **Calibration Quality**: ECE range 0.081 - 0.156

## TRAINING DATA

### Dataset Characteristics
- **Size**: 1,530 respondent-joke pairs
- **Respondents**: 51 unique individuals
- **Jokes**: 30 unique humor items
- **Collection**: Survey-based humor ratings

### Class Distribution
- **funny**: 464 (30.3%)
- **not_funny**: 958 (62.6%)
- **dont_understand**: 108 (7.1%)

### Demographic Coverage
- **Age Groups**: {'18-25': 540, '26-35': 390, '46-55': 270, '36-45': 180, '55+': 150}
- **Gender**: {'Male': 780, 'Female': 720, 'Non-binary': 30}
- **Ethnicity**: {'South Asian (e.g., Indian, Pakistani, Bangladeshi, Sri Lankan, Nepali)': 480, 'White / Caucasian': 360, 'Black / African': 150, 'Hispanic / Latino': 120, 'Middle Eastern': 90}

## EVALUATION STRATEGY

### Split Methodology
1. **Cold-Start User** (80/20): Tests generalization to new users
2. **New-Joke** (80/20): Tests generalization to new humor content  
3. **Random** (80/20): Standard ML evaluation baseline

### Performance Across Splits

- **Cold Start User**: Acc=0.687, F1-macro=0.491, ECE=0.097
- **New Joke**: Acc=0.650, F1-macro=0.311, ECE=0.156
- **Random**: Acc=0.690, F1-macro=0.544, ECE=0.081

## FAIRNESS AND BIAS ASSESSMENT

### Known Limitations
1. **Sample Size**: Limited to 51 respondents
2. **Demographic Bias**: Uneven representation across groups
3. **Cultural Context**: Survey-based collection may reflect specific cultural humor norms
4. **"dont_understand" Difficulty**: Consistently challenging class across all evaluation scenarios

### Bias Considerations
- **Gender Representation**: {'Male': 26, 'Female': 24, 'Non-binary': 1}
- **Age Distribution**: Potential skew toward certain age groups
- **Cultural Coverage**: Limited ethnicity representation may affect generalization

### Fairness Monitoring
- Per-group metrics show variation across demographic segments
- Recommend ongoing monitoring for systematic biases in production
- Consider rebalancing training data for underrepresented groups

## FEATURE ANALYSIS

### Feature Importance (Ablation Study)
- **Text Features**: 0.461 avg F1-macro
- **Demographics**: 0.288 avg F1-macro  
- **Combined**: 0.503 avg F1-macro
- **Text Advantage**: +0.173 F1-macro over demographics alone

### Key Insights
- Text features provide substantial predictive power
- Demographic features contribute meaningful signal
- Feature combination yields synergistic improvements

## USAGE GUIDELINES

### Recommended Applications
[GOOD FOR]:
- Humor content moderation systems
- Personalized content recommendation (with user consent)
- Research on humor perception patterns
- A/B testing of humor content

[USE WITH CAUTION]:
- High-stakes decisions based solely on humor predictions
- Cross-cultural applications without revalidation
- Individual psychological assessments

[NOT RECOMMENDED]:
- Medical or clinical decision-making
- Legal or employment screening
- Applications requiring >95% accuracy

### Monitoring Requirements
1. **Performance Drift**: Track F1-macro and ECE monthly
2. **Fairness Metrics**: Monitor per-group performance disparities
3. **Data Quality**: Validate input preprocessing consistency
4. **Calibration**: Reassess probability calibration quarterly

## TECHNICAL SPECIFICATIONS

### Input Requirements
- **Text**: Joke/humor content as string
- **Demographics**: age_bin, gender, ethnicity (categorical)
- **Preprocessing**: Apply saved feature encoders consistently

### Output Format  
- **Predictions**: ['funny', 'not_funny', 'dont_understand']
- **Probabilities**: Calibrated confidence scores [0,1]
- **Calibration Quality**: ECE scores for reliability assessment

## ARTIFACTS AND REPRODUCIBILITY

### Saved Components
- Preprocessing configuration and encoders
- Trained models for all evaluation splits
- Data split indices for reproducibility
- Performance metrics and calibration assessments

### Version Control
- Model trained: 2025-08-25
- scikit-learn version: 1.4.2
- Random seed: 42 (for reproducibility)

## CONTACT AND SUPPORT

For questions about model usage, limitations, or bias concerns, please refer to:
- Technical documentation in saved artifacts
- Ablation study results for feature importance
- Fairness assessment metrics for demographic considerations

---
*This model card follows responsible AI practices and should be updated with new findings.*
