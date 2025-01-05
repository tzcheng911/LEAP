library(lme4)
library(lmerTest)
library(tidyverse)
library(dplyr)
library(broom)
library(ggplot2)
library(ggpubr)
library('fastDummies')
library(effsize)
library(lsr)
library(rstatix)
library(car)

################################################################# Load the ROI data
alldata = read.csv("SSEP_roi5.csv")

# recode to collapse left and right ROIs if needed ******be careful to overwrite******
alldata = alldata %>%
  mutate(ROI =fct_collapse(ROI,Auditory = c("AuditoryL", "AuditoryR"),Motor = c("MotorL", "MotorR"),
                                  Sensory = c("SensoryL","SensoryR"),BG = c("BGL","BGR"), IFG = c("IFGL","IFGR")))
alldata = alldata %>%
  group_by(sub_id,age,condition,ROI) %>%
  summarize(X1.67Hz = mean(X1.67Hz), X1.11Hz = mean(X1.11Hz), X3.3Hz = mean(X3.3Hz),Nsubs=n_distinct(sub_id), .groups = "drop") 

# log transform to achieve normality
alldata = alldata %>% 
  mutate(log_X1.11Hz = log(X1.11Hz),log_X1.67Hz = log(X1.67Hz), log_X3.3Hz = log(X3.3Hz))

# Mutate new factors and re-level
alldata = alldata %>%
  mutate(age = as.factor(age))
alldata = alldata %>%
  mutate(condition = as.factor(condition))
alldata = alldata %>%
  mutate(ROI = as.factor(ROI))
alldata = alldata %>%
  mutate(sub_id = as.factor(sub_id))

alldata$age = relevel(alldata$age,ref="7mo")
alldata$condition = relevel(alldata$condition,ref="_02")

## Descriptive stats
# sensor
summary_alldata = alldata %>%
  group_by(age,condition,ROI) %>%
  summarize(mMeterD = mean(X1.67Hz), mMeterT = mean(X1.11Hz), mBeat = mean(X3.3Hz),Nsubs=n_distinct(sub_id), 
            seMeterD = sd(X1.67Hz)/sqrt(Nsubs),seMeterT = sd(X1.11Hz)/sqrt(Nsubs),seBeat = sd(X3.3Hz)/sqrt(Nsubs))

## 2-way Mixed effect ANOVA 
# check assumptions - outliers
outliers = alldata %>%
  group_by(condition,age) %>%
  identify_outliers(log_X1.67Hz) # change to X1.11Hz, X1.67Hz and X3.3Hz
alldata = filter(alldata, !(sub_id %in% unique(outliers$sub_id)))

# check assumptions - Normality (p > 0.05) log_X3.3Hz failed
alldata %>%
  group_by(condition,age) %>%
  shapiro_test(log_X3.3Hz) # change to X1.11Hz, X1.67Hz and X3.3Hz

# check assumptions - Homogneity of variance (p > 0.05) log_X1.67Hz failed
alldata %>%
  group_by(condition) %>%
  levene_test(log_X1.67Hz ~ age) # change to X1.11Hz, X1.67Hz and X3.3Hz

# check assumptions - Homogeneity of covariances (unless p < 0.001 and your sample sizes are unequal, ignore it)
box_m(alldata[,"log_X3.3Hz",drop=FALSE],alldata$age)

################################################################# run ANOVA
which_ROI = "Motor" # AuditoryL,AuditoryR,MotorL,MotorR,SensoryL,SensoryR,BGL,BGR,IFGL,IFGR, or the recode ROI without L/R

duple_random = filter(alldata,condition != "_04", ROI == which_ROI)
# summary(aov(log_X1.67Hz ~ age*condition+Error(sub_id/condition),data=duple_random)) # account for subj and condition effect

res.aov <- anova_test(
  data = duple_random, dv = log_X1.67Hz, wid = sub_id,
  between = age, within = condition
)

get_anova_table(res.aov)

res.aov <- anova_test(
  data = duple_random, dv = log_X3.3Hz, wid = sub_id,
  between = age, within = condition
)
get_anova_table(res.aov)

triple_random = filter(alldata,condition != "_03", ROI == which_ROI)
res.aov <- anova_test(
  data = triple_random, dv = log_X1.11Hz, wid = sub_id,
  between = age, within = condition
)
get_anova_table(res.aov)
res.aov <- anova_test(
  data = triple_random, dv = log_X3.3Hz, wid = sub_id,
  between = age, within = condition
)
get_anova_table(res.aov)

# simple main effect of age
one.way <- duple_random %>%
  group_by(condition) %>%
  anova_test(dv = log_X1.67Hz, wid = sub_id, between = age) %>%
  get_anova_table() %>%
  adjust_pvalue(method = "bonferroni")
one.way

one.way <- duple_random %>%
  group_by(condition) %>%
  anova_test(dv = log_X3.3Hz, wid = sub_id, between = age) %>%
  get_anova_table() %>%
  adjust_pvalue(method = "bonferroni")
one.way

one.way <- triple_random %>%
  group_by(condition) %>%
  anova_test(dv = log_X1.11Hz, wid = sub_id, between = age) %>%
  get_anova_table() %>%
  adjust_pvalue(method = "bonferroni")
one.way

one.way <- triple_random %>%
  group_by(condition) %>%
  anova_test(dv = log_X3.3Hz, wid = sub_id, between = age) %>%
  get_anova_table() %>%
  adjust_pvalue(method = "bonferroni")
one.way

## pairwise comparison of age and condition
# condition effect of each of the three ages
pwc <- duple_random %>%
  group_by(age) %>%
  pairwise_t_test(log_X1.67Hz ~ condition, p.adjust.method = "bonferroni")
pwc

pwc <- duple_random %>%
  group_by(age) %>%
  pairwise_t_test(log_X3.3Hz ~ condition, p.adjust.method = "bonferroni")
pwc

pwc <- triple_random %>%
  group_by(age) %>%
  pairwise_t_test(log_X1.11Hz ~ condition, p.adjust.method = "bonferroni")
pwc

pwc <- triple_random %>%
  group_by(age) %>%
  pairwise_t_test(log_X3.3Hz ~ condition, p.adjust.method = "bonferroni")
pwc

# age effect of each of the two conditions
pwc <- duple_random %>%
  group_by(condition) %>%
  pairwise_t_test(log_X1.67Hz ~ age, p.adjust.method = "bonferroni")
pwc

pwc <- duple_random %>%
  group_by(condition) %>%
  pairwise_t_test(log_X3.3Hz ~ age, p.adjust.method = "bonferroni")
pwc

pwc <- triple_random %>%
  group_by(condition) %>%
  pairwise_t_test(log_X1.11Hz ~ age, p.adjust.method = "bonferroni")
pwc

pwc <- triple_random %>%
  group_by(condition) %>%
  pairwise_t_test(log_X3.3Hz ~ age, p.adjust.method = "bonferroni")
pwc

## Visualization
ggplot(duple_random, aes(x = age, y = X1.67Hz, fill = condition)) +
  geom_bar(stat="summary", position='dodge') +
  stat_summary(fun.data=mean_se, geom="errorbar", position = position_dodge(width = 0.9), width=.1,color="grey") +
  geom_point(position = position_jitterdodge(jitter.width = 0.3,dodge.width = 0.9), color="black")+
  theme_bw()

ggplot(triple_random, aes(x = age, y = X1.11Hz, fill = condition)) +
  geom_bar(stat="summary", position='dodge') +
  stat_summary(fun.data=mean_se, geom="errorbar", position = position_dodge(width = 0.9), width=.1,color="grey") +
  geom_point(position = position_jitterdodge(jitter.width = 0.3,dodge.width = 0.9), color="black")+
  theme_bw()

ggplot(alldata, aes(x = age, y = X3.3Hz, fill = condition)) +
  geom_bar(stat="summary", position='dodge') +
  stat_summary(fun.data=mean_se, geom="errorbar", position = position_dodge(width = 0.9), width=.1,color="grey") +
  geom_point(position = position_jitterdodge(jitter.width = 0.3,dodge.width = 0.9), color="black")+
  theme_bw()

## 2-way Mixed effect lmer
# duple lmer
lmall = lmer(X1.67Hz ~ age*condition  + (1|sub_id),data= alldata,verbose=2,subset = condition != "_04")  
summary(lmall) 
lmall_noAge = lmer(X1.67Hz ~ age*condition-age:condition + (1|sub_id),data= alldata,verbose=2,subset = condition != "triple")  
anova(lmall,lmall_noAge)

# triple lmer
lmall = lmer(X1.11Hz ~ age*condition  + (1|sub_id),data= alldata,verbose=2,subset = condition != "_03")  
summary(lmall) 
lmall_noAge = lmer(X1.11Hz ~ age*condition-age:condition + (1|sub_id),data= alldata,verbose=2,subset = condition != "duple")  
anova(lmall,lmall_noAge)

## beat lmer
lmall = lmer(X3.3Hz ~ age*condition  + (1 |sub_id),data= alldata,verbose=2)  
summary(lmall) # Use early as the reference
lmall_noAge = lmer(X3.3Hz ~ age*condition-age:condition + (1|sub_id),data= alldata,verbose=2)  
anova(lmall,lmall_noAge)
